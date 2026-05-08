import pandas as pd
import torch

from .swarm import DroneSwarm, RANGE, CYLINDER_RADIUS

class Env:
    def __init__(self, blue_genes, blue_sexes, red_genes, red_sexes):
        self.device = blue_genes.device
        self.B, self.N = blue_genes.shape[:2]

        self.blue = DroneSwarm(
            blue_genes, blue_sexes,
            offset=torch.tensor([0.25, 0.25, 0])
        )

        self.red = DroneSwarm(
            red_genes, red_sexes,
            offset=torch.tensor([0.75, 0.75, 0])
        )

        self.b_kills = torch.zeros(self.B, self.N, device=self.device)
        self.b_alive_time = torch.zeros(self.B, self.N, device=self.device)
        self.r_kills = torch.zeros(self.B, self.N, device=self.device)
        self.r_alive_time = torch.zeros(self.B, self.N, device=self.device)

    def _to_df(self, batch_idx, idx, is_blue):
        kills = self.b_kills[batch_idx] if is_blue else self.r_kills[batch_idx]
        alive = self.b_alive_time[batch_idx] if is_blue else self.r_alive_time[batch_idx]

        return pd.DataFrame({
            'kills': kills[idx].cpu().numpy(),
            'lifespan': alive[idx].cpu().numpy()
        }, index=idx.cpu().tolist())

    def get_stats(self, batch_idx=0, top_k=None):
        if top_k is None:
            top_k = self.N

        # Slice out the specific batch before sorting!
        b_survivors = self.b_alive_time[batch_idx].sort(descending=True).indices[:top_k]
        b_killers = self.b_kills[batch_idx].sort(descending=True).indices[:top_k]
        r_survivors = self.r_alive_time[batch_idx].sort(descending=True).indices[:top_k]
        r_killers = self.r_kills[batch_idx].sort(descending=True).indices[:top_k]

        return (
            self._to_df(batch_idx, b_survivors, True),
            self._to_df(batch_idx, b_killers, True),
            self._to_df(batch_idx, r_survivors, False),
            self._to_df(batch_idx, r_killers, False)
        )

    def update(self, viz=False):
        b_killed, r_killed, b_new_kills, r_new_kills = self.attack()

        self.b_alive_time += self.blue.alive.float()
        self.r_alive_time += self.red.alive.float()
        self.b_kills += b_new_kills
        self.r_kills += r_new_kills

        red_s = self.red.s.clone()
        blue_s = self.blue.s.clone()

        blue_loss, b_collisions = self.blue.update(red_s, b_killed)
        red_loss, r_collisions = self.red.update(blue_s, r_killed)
        game_over = torch.stack([blue_loss, red_loss], dim=1)

        if viz:
            # We strictly pull from batch index 0 for visualization
            explosion_pos = torch.cat([blue_s[0, b_killed[0]], red_s[0, r_killed[0]]], dim=0)

            # Use the boolean masks to pull the coordinates of the crash sites
            b_col_coords = blue_s[0, b_collisions[0]]
            r_col_coords = red_s[0, r_collisions[0]]

            return game_over, explosion_pos, b_col_coords, r_col_coords

        # B x 2
        return game_over

    def attack(self):
        b_speed = torch.norm(self.blue.v, dim=-1, keepdim=True).clamp(min=1e-5)
        b_heading = (self.blue.v / b_speed).unsqueeze(2)

        r_speed = torch.norm(self.red.v, dim=-1, keepdim=True).clamp(min=1e-5)
        r_heading = (self.red.v / r_speed).unsqueeze(2)

        r_killed, b_new_kills = self._calc_hits(self.blue.s, self.red.s, b_heading, self.blue.alive, self.red.alive)
        b_killed, r_new_kills = self._calc_hits(self.red.s, self.blue.s, r_heading, self.red.alive, self.blue.alive)

        return b_killed, r_killed, b_new_kills, r_new_kills

    def _calc_hits(self, attacker_s, defender_s, attacker_heading, a_alive, d_alive):
        # (B, 1, M, 3) - (B, N, 1, 3) -> (B, N, M, 3)
        d_a2d = defender_s.unsqueeze(1) - attacker_s.unsqueeze(2)
        forward_a2d = (d_a2d * attacker_heading).sum(dim=-1)

        perp_vec_a2d = d_a2d - forward_a2d.unsqueeze(-1) * attacker_heading
        perp_dist_a2d = perp_vec_a2d.norm(dim=-1)

        # Valid hit: In front, in range, inside cylinder, AND both parties are alive
        hits = (forward_a2d > 0) & (forward_a2d <= RANGE) & (perp_dist_a2d <= CYLINDER_RADIUS)
        hits = hits & a_alive.unsqueeze(2) & d_alive.unsqueeze(1)

        killed = hits.any(dim=1)
        kills_per_attacker = hits.float().sum(dim=2)

        return killed, kills_per_attacker
