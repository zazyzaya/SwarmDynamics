import torch

from src.dna import Genes
from src.swarm import DroneSwarm
from src.phys_globals import CYLINDER_RADIUS, RANGE

class Env:
    def __init__(self, blue_swarm: DroneSwarm, red_swarm: DroneSwarm,
                 blue_shape, red_shape, device, obstacles):
        self.device = device

        self.blue = blue_swarm
        self.red = red_swarm

        self.b_kills = torch.zeros(*blue_shape, device=self.device)
        self.b_alive_time = torch.zeros(*blue_shape, device=self.device)

        self.r_kills = torch.zeros(*red_shape, device=self.device)
        self.r_alive_time = torch.zeros(*red_shape, device=self.device)

        # (B) x N x 3 x 2 tensor of triangle coords, (B) x N x 1 tensor of triangle heights
        self.obstacles = obstacles

    def attack(self):
        # dim=-1 works for both (N, 3) and (B, N, 3)
        b_speed = torch.norm(self.blue.v, dim=-1, keepdim=True).clamp(min=1e-5)
        b_heading = (self.blue.v / b_speed).unsqueeze(-2)

        r_speed = torch.norm(self.red.v, dim=-1, keepdim=True).clamp(min=1e-5)
        r_heading = (self.red.v / r_speed).unsqueeze(-2)

        # 1. Extract the CAN_FIRE gene mask for both teams (Works for both 2D and 3D!)
        b_can_fire = self.blue.genes[..., Genes.CAN_FIRE] > 0
        r_can_fire = self.red.genes[..., Genes.CAN_FIRE] > 0

        # 2. Pass the fire masks into _calc_hits
        r_killed, b_new_kills = self._calc_hits(
            self.blue.s, self.red.s, b_heading,
            self.blue.alive, self.red.alive, b_can_fire
        )

        b_killed, r_new_kills = self._calc_hits(
            self.red.s, self.blue.s, r_heading,
            self.red.alive, self.blue.alive, r_can_fire
        )

        return b_killed, r_killed, b_new_kills, r_new_kills

    def crashes(self):
        if self.obstacles[0].size(-2):
            b_crashes = self.blue.column_collisions(*self.obstacles)
            r_crashes = self.red.column_collisions(*self.obstacles)
            return b_crashes, r_crashes
        else:
            return \
                torch.zeros(self.blue.s.shape[:-1], device=self.device, dtype=torch.bool), \
                torch.zeros(self.red.s.shape[:-1], device=self.device, dtype=torch.bool)

    def _calc_hits(self, attacker_s, defender_s, attacker_heading, a_alive, d_alive, a_can_fire):
        # GPU: (B, 1, M, 3) - (B, N, 1, 3) -> (B, N, M, 3)
        # CPU: (1, M, 3) - (N, 1, 3) -> (N, M, 3)
        d_a2d = defender_s.unsqueeze(-3) - attacker_s.unsqueeze(-2)
        forward_a2d = (d_a2d * attacker_heading).sum(dim=-1)

        perp_vec_a2d = d_a2d - forward_a2d.unsqueeze(-1) * attacker_heading
        perp_dist_a2d = perp_vec_a2d.norm(dim=-1)

        # Valid hit: In front, in range, inside cylinder
        hits = (forward_a2d > 0) & (forward_a2d <= RANGE) & (perp_dist_a2d <= CYLINDER_RADIUS)

        # AND both parties are alive, AND the attacker is genetically allowed to shoot
        # -1 unsqueezes the Attacker dim (N), -2 unsqueezes the Defender dim (M)
        hits = hits & a_alive.unsqueeze(-1) & d_alive.unsqueeze(-2) & a_can_fire.unsqueeze(-1)

        # Dim -2 is always the Attackers, Dim -1 is always the Defenders
        killed = hits.any(dim=-2)
        kills_per_attacker = hits.float().sum(dim=-1)

        return killed, kills_per_attacker