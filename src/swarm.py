import torch
from src.dna import NUM_SEXES, Genes
from src.phys_globals import *

class DroneSwarm:
    def __init__(self, genes, sexes, shape, offset):
        """
        A single unified init that handles both (N,) and (B, N) shapes!
        """
        self.device = genes.device
        self.genes = genes
        self.shape = shape

        # Spawn locations
        self.s = self._starting_loc(shape, offset)

        # Velocity
        self.v = (torch.rand(*shape, 3, device=self.device) - 0.5) * 2

        # Alive mask
        self.alive = torch.ones(*shape, dtype=torch.bool, device=self.device)

        # Fast, dimension-agnostic one-hot encoding for sexes
        self.sexes = torch.zeros(*shape, NUM_SEXES, device=self.device)
        self.sexes.scatter_(-1, sexes.unsqueeze(-1), 1)

        # ID tracking
        self.id = torch.arange(shape[-1], device=self.device).expand(shape)

    def _starting_loc(self, shape, offset):
        # Spawns (N, 3) or (B, N, 3) tensors natively! No list comprehension needed.
        s = (torch.rand(*shape, 3, device=self.device) - 0.5) * 0.5
        s[..., Z] = 1.0
        s += offset.to(self.device)
        return s

    def physics_clipping(self, tot_dv):
        steering = tot_dv - self.v
        steering_mag = torch.norm(steering, dim=-1, keepdim=True)
        clamped_steering = steering * (torch.clamp(steering_mag, max=MAX_TURN_FORCE) / (steering_mag + 1e-8))
        self.v = self.v + clamped_steering

        speed = torch.norm(self.v, dim=-1, keepdim=True)
        clamped_speed = torch.clamp(speed, min=MIN_SPEED, max=MAX_SPEED)
        self.v = self.v * (clamped_speed / (speed + 1e-8))

    def column_collisions(self, vertices, heights):
        """
        Calculates if drones have crashed into ANY of the T flat-topped triangular columns.
            vertices: Tensor of shape ((B,) T, 3, 2)
            heights: Tensor of shape (T,) or (T, 1)
        """
        if vertices is None or vertices.numel() == 0:
            return torch.zeros(self.s.shape[:-1], dtype=torch.bool, device=self.device)

        # 1. The Height Check
        # Drone Z shape: (..., N, 1) | Heights shape: (..., 1, T)
        in_z = self.s[..., Z].unsqueeze(-1) < heights.unsqueeze(-2)

        # 2. The 2D Cross Product Test
        # Drone XY shape: (..., N, 1, 2)
        p = self.s[..., :2].unsqueeze(-2)

        # Extract the T vertices. Shapes are (..., T, 2)
        v0, v1, v2 = vertices[..., 0, :], vertices[..., 1, :], vertices[..., 2, :]

        # Edge vectors of the T triangles: (..., T, 2)
        e0 = v1 - v0
        e1 = v2 - v1
        e2 = v0 - v2

        # Vectors from vertices to drones.
        # (..., N, 1, 2) - (..., 1, T, 2) -> PyTorch aligns the dimensions!
        p0 = p - v0.unsqueeze(-3)
        p1 = p - v1.unsqueeze(-3)
        p2 = p - v2.unsqueeze(-3)

        # Calculate 2D Cross Products
        # Unpack edges and unsqueeze so (..., T) becomes (..., 1, T)
        e0x, e0y = e0[..., 0].unsqueeze(-2), e0[..., 1].unsqueeze(-2)
        e1x, e1y = e1[..., 0].unsqueeze(-2), e1[..., 1].unsqueeze(-2)
        e2x, e2y = e2[..., 0].unsqueeze(-2), e2[..., 1].unsqueeze(-2)

        c0 = (e0x * p0[..., 1]) - (e0y * p0[..., 0])
        c1 = (e1x * p1[..., 1]) - (e1y * p1[..., 0])
        c2 = (e2x * p2[..., 1]) - (e2y * p2[..., 0])

        # Check if inside the footprint
        inside_triangle = ((c0 >= 0) & (c1 >= 0) & (c2 >= 0)) | ((c0 <= 0) & (c1 <= 0) & (c2 <= 0))

        # Combine height and footprint checks
        crashes = inside_triangle & in_z

        # Any collision is a death
        return crashes.any(dim=-1)

    def boid(self, other_s, obs_verts, obs_heights):
        """
        The massive unified Boid function. Using negative dims (-1, -2, -3) allows
        these matrices to calculate distance, sum, and multiply completely agnostic
        to whether they are 2D (CPU) or 3D (GPU batched).
        """
        dist = torch.norm(self.s.unsqueeze(-2) - self.s.unsqueeze(-3), dim=-1)

        # Check for friendly collisions
        eye = torch.eye(self.s.size(-2), device=self.device).bool()
        dist.masked_fill_(eye, float('inf'))
        collisions = (dist < COLLISION_DIST).sum(dim=-1).bool() & self.alive

        # Only update living entities
        alive_mask = self.alive.unsqueeze(-1) & self.alive.unsqueeze(-2)

        # Step 1: Uncrowd the boids
        too_close = (dist < self.genes[..., Genes.PROTECTED:Genes.PROTECTED+1]) & alive_mask
        close_count = too_close.sum(dim=-1, keepdim=True)
        dv = too_close.float() @ self.s
        close_dv = (self.s * close_count - dv) * self.genes[..., Genes.ALPHA:Genes.ALPHA+1]

        # Step 1.5: Find neighbors
        visible = (dist < COMM_RANGE) & alive_mask
        neighbors = visible.sum(dim=-1, keepdim=True)
        has_neighbors = (neighbors > 0).float()

        # Step 2: Align the boids
        dv = visible.float() @ self.v
        dv /= neighbors.clamp(min=1)
        align_dv = (dv - self.v) * has_neighbors * self.genes[..., Genes.BETA:Genes.BETA+1]

        # Step 3: Cohesion
        dv = visible.float() @ self.s
        dv /= neighbors.clamp(min=1)
        cohesion_dv = (dv - self.s) * has_neighbors * self.genes[..., Genes.GAMMA:Genes.GAMMA+1]

        # Step 4: Boundary avoidance
        min_bounds = torch.full_like(self.s, TURN_MARGIN)
        min_bounds[..., Z] = self.genes[..., Genes.MARGIN]

        max_bounds = torch.full_like(self.s, 1.0 - TURN_MARGIN)
        max_bounds[..., Z] = CEILING - TURN_MARGIN

        too_low = (min_bounds - self.s).clamp(min=0)
        too_high = (self.s - max_bounds).clamp(min=0)

        turn_dv = (too_low - too_high) * TURN_FACTOR
        turn_dv[..., Z] = (too_low[..., Z] * self.genes[..., Genes.TURN]) - (too_high[..., Z] * TURN_FACTOR)

        tot_dv = self.v + close_dv + align_dv + cohesion_dv + turn_dv

        if obs_verts is not None and obs_heights is not None:
            # Calculate the centers of the T columns: Shape (..., T, 2)
            obs_centers = obs_verts.mean(dim=-2)

            # Extract drone XY: (..., N, 1, 2)
            p = self.s[..., :2].unsqueeze(-2)

            # Vector pointing from obstacle centers to drones: (..., N, T, 2)
            push_xy = p - obs_centers.unsqueeze(-3)
            dist_xy = torch.norm(push_xy, dim=-1)

            # Are they too close AND below the roof? (..., N, T)
            in_z_danger = self.s[..., Z].unsqueeze(-1) < (obs_heights.unsqueeze(-2) + AVOID_MARGIN_Z)
            danger = (dist_xy < AVOID_MARGIN_XY) & in_z_danger & self.alive.unsqueeze(-1)

            # Calculate push strength (closer = exponentially stronger)
            push_strength = (AVOID_MARGIN_XY - dist_xy).clamp(min=0) / AVOID_MARGIN_XY
            push_strength = push_strength * danger.float()

            # Normalize direction
            push_dir = push_xy / (dist_xy.unsqueeze(-1) + 1e-8)

            # Sum the forces pushing away from all T obstacles
            obs_dv_xy = (push_dir * push_strength.unsqueeze(-1)).sum(dim=-2)

            # Add a slight upward thrust to help them fly over
            obs_dv_z = push_strength.sum(dim=-1).unsqueeze(-1)

            # Combine into 3D vector and apply TURN_FACTOR multiplier
            obs_dv = torch.cat([obs_dv_xy, obs_dv_z], dim=-1)
            obs_dv = obs_dv * self.genes[..., Genes.OBSTACLE_AVOID].unsqueeze(-1) * TURN_FACTOR * 5.0

            tot_dv += obs_dv

        # Step 5: Enemy Interaction
        if other_s is not None and other_s.numel() > 0:
            enemy_dist = torch.norm(self.s.unsqueeze(-2) - other_s.unsqueeze(-3), dim=-1)
            other_alive = (other_s[..., Z] > -0.0).unsqueeze(-2)
            self_alive = self.alive.unsqueeze(-1)

            enemy_collisions = ((enemy_dist < COLLISION_DIST) & self_alive & other_alive).sum(dim=-1).bool()
            collisions = collisions | enemy_collisions

            enemy_visible = (enemy_dist < COMM_RANGE) & self_alive & other_alive
            enemy_counts = enemy_visible.sum(dim=-1, keepdim=True)
            has_enemies = (enemy_counts > 0).float()

            enemy_sum = enemy_visible.float() @ other_s
            enemy_center = enemy_sum / enemy_counts.clamp(min=1)

            desire_dv = (enemy_center - self.s) * has_enemies * self.genes[..., Genes.DESIRE:Genes.DESIRE+1]
            fear_dv = (self.s - enemy_center) * has_enemies * self.genes[..., Genes.FEAR:Genes.FEAR+1]
            tot_dv += desire_dv + fear_dv

            # Step 6: Send intel to others
            valid_senders = self.sexes * has_enemies
            broadcast = valid_senders.unsqueeze(-1) * enemy_center.unsqueeze(-2)

            global_msg_sum = broadcast.sum(dim=-3)
            global_msg_counts = valid_senders.sum(dim=-2)
            msg = global_msg_sum / global_msg_counts.clamp(min=1).unsqueeze(-1)

            listen_prefs = self.genes[..., Genes.LISTEN_BIAS : Genes.LISTEN_BIAS + NUM_SEXES]
            has_signal = (global_msg_counts > 0).float().unsqueeze(-2)
            active_listen_prefs = listen_prefs * has_signal

            channel_dv = msg.unsqueeze(-3) - self.s.unsqueeze(-2)
            radio_dv = (active_listen_prefs.unsqueeze(-1) * channel_dv).sum(dim=-2)
            radio_dv *= self.genes[..., Genes.LISTEN_BIAS_VAL : Genes.LISTEN_BIAS_VAL + 1]
            tot_dv += radio_dv

        # Step 7: Follow biased members
        local_clan_counts = visible.float() @ self.sexes
        clan_v_split = self.sexes.unsqueeze(-1) * self.v.unsqueeze(-2)

        # Flatten handles arbitrary dimensions cleanly for matrix multiplication
        flat_v = clan_v_split.flatten(start_dim=-2)
        local_clan_v_sum = (visible.float() @ flat_v).unflatten(-1, (NUM_SEXES, 3))

        local_clan_v_avg = local_clan_v_sum / local_clan_counts.clamp(min=1).unsqueeze(-1)

        preferences = self.genes[..., Genes.FOLLOW_BIAS:Genes.FOLLOW_BIAS + NUM_SEXES]
        visible_sexes_mask = (local_clan_counts > 0).float()
        active_preferences = preferences * visible_sexes_mask

        preferred_v = (active_preferences.unsqueeze(-1) * local_clan_v_avg).sum(dim=-2)
        bias_weight = self.genes[..., Genes.FOLLOW_BIAS_VAL:Genes.FOLLOW_BIAS_VAL+1]
        tot_dv += preferred_v * bias_weight

        self.physics_clipping(tot_dv)

        return collisions