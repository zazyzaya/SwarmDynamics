import pandas as pd
import torch

from src.dna import Genes
from .swarm import DroneSwarm

RANGE = 0.1
CYLINDER_RADIUS = 0.01

class Env:
    def __init__(self, blue_genes, blue_sexes, red_genes, red_sexes):
        self.device = blue_genes.device

        self.blue = DroneSwarm(
            blue_genes, blue_sexes,
            offset=torch.tensor([0.25, 0.25, 0]).to(self.device),
        )

        self.red = DroneSwarm(
            red_genes, red_sexes,
            offset=torch.tensor([0.75, 0.75, 0]).to(self.device),
        )

        self.n_blue = blue_genes.size(0)
        self.n_red = red_genes.size(0)

        self.b_kills = torch.zeros(self.n_blue).to(self.device)
        self.b_alive_time = torch.zeros(self.n_blue).to(self.device)
        self.r_kills = torch.zeros(self.n_red).to(self.device)
        self.r_alive_time = torch.zeros(self.n_red).to(self.device)

    def _to_df(self, idx, is_blue):
        kills = self.b_kills if is_blue else self.r_kills
        alive = self.b_alive_time if is_blue else self.r_alive_time

        return pd.DataFrame({
            'kills': kills[idx].cpu(),
            'lifespan': alive[idx].cpu()
        }, index=idx.cpu().tolist())

    def get_stats(self, top_k=None):
        if top_k is None:
            top_k = max(self.n_blue, self.n_red)

        b_survivors = self.b_alive_time.sort(descending=True).indices[:top_k]
        b_killers = self.b_kills.sort(descending=True).indices[:top_k]
        r_survivors = self.r_alive_time.sort(descending=True).indices[:top_k]
        r_killers = self.r_kills.sort(descending=True).indices[:top_k]

        return (
            self._to_df(b_survivors, True),
            self._to_df(b_killers, True),
            self._to_df(r_survivors, False),
            self._to_df(r_killers, False)
        )

    def update(self, viz=False):
        b_killed, r_killed, b_new_kills, r_new_kills = self.attack()
        explosion_pos = torch.cat([self.blue.s[b_killed], self.red.s[r_killed]])

        self.b_alive_time[self.blue.id] += 1
        self.r_alive_time[self.red.id] += 1

        self.b_kills[self.blue.id] += b_new_kills
        self.r_kills[self.red.id] += r_new_kills

        red_s = self.red.s.clone()
        blue_s = self.blue.s.clone()

        blue_loss, b_collisions = self.blue.update(red_s, b_killed)
        red_loss, r_collisions = self.red.update(blue_s, r_killed)

        if viz:
            return torch.tensor([blue_loss, red_loss]), explosion_pos, b_collisions, r_collisions
        else:
            return torch.tensor([blue_loss, red_loss])

    def attack(self):
        b_speed = torch.norm(self.blue.v, dim=1, keepdim=True).clamp(min=1e-5)
        b_heading = (self.blue.v / b_speed).unsqueeze(1)

        r_speed = torch.norm(self.red.v, dim=1, keepdim=True).clamp(min=1e-5)
        r_heading = (self.red.v / r_speed).unsqueeze(1)

        # 1. Identify which drones are genetically allowed to fire
        b_can_fire = self.blue.genes[:, Genes.CAN_FIRE] > 0
        r_can_fire = self.red.genes[:, Genes.CAN_FIRE] > 0

        # 2. Initialize tracking tensors for the whole swarm
        b_new_kills = torch.zeros(self.blue.n, device=self.blue.s.device)
        r_new_kills = torch.zeros(self.red.n, device=self.red.s.device)

        r_killed = torch.zeros(self.red.n, dtype=torch.bool, device=self.red.s.device)
        b_killed = torch.zeros(self.blue.n, dtype=torch.bool, device=self.blue.s.device)

        # 3. Calculate Blue attacks (only if at least one Blue can fire)
        if b_can_fire.any():
            r_hit_mask, b_firing_kills = self._calc_hits(
                self.blue.s[b_can_fire],
                self.red.s,
                b_heading[b_can_fire]
            )
            r_killed = r_killed | r_hit_mask
            # Map the kills back to the specific Blues that fired
            b_new_kills[b_can_fire] = b_firing_kills

        # 4. Calculate Red attacks (only if at least one Red can fire)
        if r_can_fire.any():
            b_hit_mask, r_firing_kills = self._calc_hits(
                self.red.s[r_can_fire],
                self.blue.s,
                r_heading[r_can_fire]
            )
            b_killed = b_killed | b_hit_mask
            # Map the kills back to the specific Reds that fired
            r_new_kills[r_can_fire] = r_firing_kills

        return b_killed, r_killed, b_new_kills, r_new_kills

    def _calc_hits(self, attacker_s, defender_s, attacker_heading):
        # Distance from each blue to each red: (N_blue, M_red, 3)
        d_a2d = defender_s.unsqueeze(0) - attacker_s.unsqueeze(1)

        # Project D onto Blue's heading to get Forward Distance
        forward_a2d = (d_a2d * attacker_heading).sum(dim=-1)

        # Subtract the forward projection from D to get the Perpendicular vector
        perp_vec_a2d = d_a2d - forward_a2d.unsqueeze(-1) * attacker_heading
        perp_dist_a2d = perp_vec_a2d.norm(dim=-1)

        # Valid hit: In front (>0), within range, and inside cylinder radius
        hits = (forward_a2d > 0) & (forward_a2d <= RANGE) & (perp_dist_a2d <= CYLINDER_RADIUS)
        killed = hits.any(dim=0) # Red is killed if ANY blue hit it
        kills_per_attacker = hits.float().sum(dim=1)

        return killed, kills_per_attacker