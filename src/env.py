import torch

from src.dna import Genes
from src.swarm import DroneSwarm
from src.phys_globals import CYLINDER_RADIUS, FIRING_RANGE

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
        hits = (forward_a2d > 0) & (forward_a2d <= FIRING_RANGE) & (perp_dist_a2d <= CYLINDER_RADIUS)

        # AND both parties are alive, AND the attacker is genetically allowed to shoot
        # -1 unsqueezes the Attacker dim (N), -2 unsqueezes the Defender dim (M)
        hits = hits & a_alive.unsqueeze(-1) & d_alive.unsqueeze(-2) & a_can_fire.unsqueeze(-1)

        # 2. Line-of-Sight (Raycasting) Check
        if self.obstacles[0].numel() > 0:
            obs_centers, obs_heights, obs_radii = self.obstacles

            # Unsqueeze everything into a 4D matrix: (..., Attackers, Defenders, Obstacles, Dims)
            # A shape: (..., N, 1, 1, 3)
            # D shape: (..., 1, M, 1, 3)
            # C shape: (..., 1, 1, T, 2)
            A = attacker_s.unsqueeze(-2).unsqueeze(-2)
            D = defender_s.unsqueeze(-3).unsqueeze(-2)
            C = obs_centers.unsqueeze(-3).unsqueeze(-3)

            A_2d, D_2d = A[..., :2], D[..., :2]

            # Vector from Attacker to Defender
            V = D_2d - A_2d
            # Vector from Attacker to Obstacle Center
            W = C - A_2d

            # Find the scalar projection of W onto V (t is the percentage along the line)
            v_sq = (V * V).sum(dim=-1)
            t = (W * V).sum(dim=-1) / (v_sq + 1e-8)

            # Clamp t between 0.0 and 1.0 to restrict it to the line segment between the two drones
            t = t.clamp(min=0.0, max=1.0)

            # Find the closest physical 2D point on the line segment to the obstacle
            closest_p = A_2d + t.unsqueeze(-1) * V

            # Is that closest point inside the column's radius?
            dist_sq = ((closest_p - C)**2).sum(dim=-1)
            radii = obs_radii.unsqueeze(-2).unsqueeze(-2)
            hit_xy = dist_sq <= (radii ** 2)

            # Interpolate the Z height of the laser at point `t`
            A_z, D_z = A[..., 2], D[..., 2]
            laser_z = A_z + t * (D_z - A_z)

            # Is the laser lower than the roof of the column?
            heights = obs_heights.unsqueeze(-2).unsqueeze(-2)
            hit_z = laser_z < heights

            # A shot is blocked if it hits the XY radius AND is below the Z roof
            blocked_by_obs = hit_xy & hit_z

            # If ANY obstacle blocks the line of sight, the shot is invalid
            blocked_by_any = blocked_by_obs.any(dim=-1)

            hits = hits & ~blocked_by_any

        # Dim -2 is always the Attackers, Dim -1 is always the Defenders
        killed = hits.any(dim=-2)
        kills_per_attacker = hits.float().sum(dim=-1)

        return killed, kills_per_attacker