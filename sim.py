import pandas as pd
import torch

from drone_team import DroneFlock, NUM_CLANS

RANGE = 0.1
CYLINDER_RADIUS = 0.01

class Env:
    def __init__(self, num_blue, num_red):
        self.blue = DroneFlock(
            num_blue,
            torch.randint(0, NUM_CLANS, (num_blue,)),
            offset=torch.tensor([0.25, 0.25, 0])
        )

        self.red = DroneFlock(
            num_red,
            torch.randint(0, NUM_CLANS, (num_red,)),
            offset=torch.tensor([0.75, 0.75, 0])
        )

        self.n_blue = num_blue
        self.n_red = num_red

        self.b_kills = torch.zeros(self.n_blue)
        self.b_alive_time = torch.zeros(self.n_blue)
        self.r_kills = torch.zeros(self.n_red)
        self.r_alive_time = torch.zeros(self.n_red)

    def _to_df(self, idx, is_blue):
        kills = self.b_kills if is_blue else self.r_kills
        alive = self.b_alive_time if is_blue else self.r_alive_time

        return pd.DataFrame({
            'kills': kills[idx],
            'lifespan': alive[idx]
        }, index=idx.tolist())

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

    def update(self):
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

        return torch.tensor([blue_loss, red_loss]), explosion_pos, b_collisions, r_collisions

    def attack(self):
        b_speed = torch.norm(self.blue.v, dim=1, keepdim=True).clamp(min=1e-5)
        b_heading = (self.blue.v / b_speed).unsqueeze(1)

        r_speed = torch.norm(self.red.v, dim=1, keepdim=True).clamp(min=1e-5)
        r_heading = (self.red.v / r_speed).unsqueeze(1)

        r_killed, b_new_kills = self._calc_hits(self.blue.s, self.red.s, b_heading)
        b_killed, r_new_kills = self._calc_hits(self.red.s, self.blue.s, r_heading)

        return b_killed, r_killed, b_new_kills, r_new_kills

    def _calc_hits(self, attacker_s, defender_s, attacker_heading):
        # Distance from each blue to each red: (N_blue, M_red, 3)
        D_a2d = defender_s.unsqueeze(0) - attacker_s.unsqueeze(1)

        # Project D onto Blue's heading to get Forward Distance
        forward_a2d = (D_a2d * attacker_heading).sum(dim=-1)

        # Subtract the forward projection from D to get the Perpendicular vector
        perp_vec_a2d = D_a2d - forward_a2d.unsqueeze(-1) * attacker_heading
        perp_dist_a2d = perp_vec_a2d.norm(dim=-1)

        # Valid hit: In front (>0), within range, and inside cylinder radius
        hits = (forward_a2d > 0) & (forward_a2d <= RANGE) & (perp_dist_a2d <= CYLINDER_RADIUS)
        killed = hits.any(dim=0) # Red is killed if ANY blue hit it
        kills_per_attacker = hits.float().sum(dim=1)

        return killed, kills_per_attacker

    def coords(self):
        return (
            self.blue.s[self.blue.alive],
            self.red.s[self.red.alive]
        )