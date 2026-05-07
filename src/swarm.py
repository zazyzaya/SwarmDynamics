import torch

from .dna import NUM_SEXES, Genes

DELTA_T = 0.002

X=0; Y=1; Z=2
CEILING = 5.
TURN_FACTOR = 1.5
TURN_MARGIN = 0.1

RANGE = 0.1
CYLINDER_RADIUS = 0.01
COMM_RANGE = 0.15 # Arbitrary
MAX_SPEED = 6
MIN_SPEED = 2
COLLISION_DIST = 0.01

class DroneSwarm:
    def __init__(self, genes, sexes, offset=torch.tensor([0,0,0])):
        self.B, self.N = genes.shape[:2]
        self.device = genes.device

        self.s = torch.stack([
            self.__starting_loc(offset)
            for _ in range(self.N)
        ], dim=1) # B x N x 3

        self.v = (torch.rand(self.B, self.N, 3, device=self.device) - 0.5) * 2
        self.alive = torch.ones(self.B, self.N, dtype=torch.bool, device=self.device)

        self.genes = genes
        self.sexes = torch.zeros((self.B, self.N, NUM_SEXES), device=self.device)

        b_idx = torch.arange(self.B).view(-1, 1).expand(self.B, self.N)
        n_idx = torch.arange(self.N).view(1, -1).expand(self.B, self.N)
        self.sexes[b_idx, n_idx, sexes] = 1

        self.id = torch.arange(self.N).to(self.device).expand(self.B, self.N)

    def __starting_loc(self, offset=torch.zeros(3)):
        s = (torch.rand(self.B, 3, device=self.device) - 0.5) * 0.5
        s[:, Z] = 1
        s += offset.to(self.device)
        return s

    def update(self, other_s, killed=None):
        collisions = self.boid(other_s)
        self.s += self.v * DELTA_T

        if killed is None:
            killed = torch.zeros_like(collisions)

        dead = (self.s[..., Z] <= 0).logical_or(collisions).logical_or(killed)
        self.alive = self.alive & ~dead

        # "Bury" the dead so they don't mess w phys
        self.v[~self.alive] = 0.0
        self.s[..., Z][~self.alive] = -100.0

        team_lost = (~self.alive).sum(dim=1) == self.N
        return team_lost, collisions

    def boid(self, other_s):
        dist = torch.norm(self.s.unsqueeze(2) - self.s.unsqueeze(1), dim=-1)

        # Check for friendly collisions
        eye = torch.eye(self.N, device=self.device).bool().unsqueeze(0)
        dist.masked_fill_(eye, float('inf'))
        collisions = (dist < COLLISION_DIST).sum(dim=2).bool() & self.alive

        # Only update living entities
        alive_mask = self.alive.unsqueeze(1) & self.alive.unsqueeze(2)

        # Step 1: Uncrowd the boids
        too_close = (dist < self.genes[..., Genes.PROTECTED:Genes.PROTECTED+1]) & alive_mask
        close_count = too_close.sum(dim=-1, keepdim=True)
        dv = too_close.float() @ self.s # Add positions of birds that are too close
        close_dv = (self.s*close_count - dv) * self.genes[..., Genes.ALPHA:Genes.ALPHA+1] # And move the boids away from them

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
        min_bounds = self.s.new_tensor([TURN_MARGIN, TURN_MARGIN, 0]).view(1,1,3).expand(self.B, self.N, 3).clone()
        min_bounds[..., Z] = self.genes[..., Genes.MARGIN].squeeze(-1)
        max_bounds = self.s.new_tensor([1.0, 1.0, CEILING]) - TURN_MARGIN

        # Calculate penetration depth into the margins
        too_low = (min_bounds - self.s).clamp(min=0)
        too_high = (self.s - max_bounds).clamp(min=0)

        # Apply standard TURN factor to all axes (Walls and Ceiling)
        turn_dv = (too_low - too_high) * TURN_FACTOR
        turn_dv[..., Z] = (too_low[..., Z] * self.genes[..., Genes.TURN].squeeze(-1)) - (too_high[..., Z] * TURN_FACTOR)

        tot_dv = self.v + close_dv + align_dv + cohesion_dv + turn_dv

        # Step 5: Enemy Interaction
        if other_s is not None and other_s.numel() > 0:
            # (B, N, 1, 3) - (B, 1, M, 3) -> (B, N, M, 3)
            enemy_dist = torch.norm(self.s.unsqueeze(2) - other_s.unsqueeze(1), dim=-1)
            other_alive = (other_s[..., Z] > -0.0).unsqueeze(1)
            self_alive = self.alive.unsqueeze(2)

            enemy_collisions = ((enemy_dist < COLLISION_DIST) & self_alive & other_alive).sum(dim=-1).bool()
            collisions = collisions | enemy_collisions

            # Enemies are only visible if BOTH are alive
            enemy_visible = (enemy_dist < COMM_RANGE) & self_alive & other_alive
            enemy_counts = enemy_visible.sum(dim=-1, keepdim=True)
            has_enemies = (enemy_counts > 0).float()

            # Calculate the center of mass of visible enemies
            enemy_sum = enemy_visible.float() @ other_s
            enemy_center = enemy_sum / enemy_counts.clamp(min=1)

            # Calculate forces
            desire_dv = (enemy_center - self.s) * has_enemies * self.genes[..., Genes.DESIRE:Genes.DESIRE+1]
            fear_dv = (self.s - enemy_center) * has_enemies * self.genes[..., Genes.FEAR:Genes.FEAR+1]
            tot_dv += desire_dv + fear_dv

            # Step 6: Send intel to others
            valid_senders = self.sexes * has_enemies
            broadcast = valid_senders.unsqueeze(-1) * enemy_center.unsqueeze(2) # B x N x sexes x 3

            # Assume drones can communicate with all others
            global_msg_sum = broadcast.sum(dim=1)
            global_msg_counts = valid_senders.sum(dim=1)
            msg = global_msg_sum / global_msg_counts.clamp(min=1).unsqueeze(-1)

            # Respond to received messages
            listen_prefs = self.genes[..., Genes.LISTEN_BIAS : Genes.LISTEN_BIAS + NUM_SEXES]
            has_signal = (global_msg_counts > 0).float().unsqueeze(1)
            active_listen_prefs = listen_prefs * has_signal

            channel_dv = msg.unsqueeze(1) - self.s.unsqueeze(2)
            radio_dv = (active_listen_prefs.unsqueeze(-1) * channel_dv).sum(dim=2)
            radio_dv *= self.genes[..., Genes.LISTEN_BIAS_VAL : Genes.LISTEN_BIAS_VAL + 1]
            tot_dv += radio_dv

        # Step 7: Follow biased members
        local_clan_counts = visible.float() @ self.sexes

        # Map velocities into their clan channels: (N, C, 3)
        clan_v_split = self.sexes.unsqueeze(-1) * self.v.unsqueeze(2)
        local_clan_v_sum = visible.float() @ clan_v_split.view(self.B, self.N, NUM_SEXES * 3)
        local_clan_v_sum = local_clan_v_sum.view(self.B, self.N, NUM_SEXES, 3)
        local_clan_v_avg = local_clan_v_sum / local_clan_counts.clamp(min=1).unsqueeze(-1)

        preferences = self.genes[..., Genes.FOLLOW_BIAS:Genes.FOLLOW_BIAS + NUM_SEXES]
        visible_sexes_mask = (local_clan_counts > 0).float()
        active_preferences = preferences * visible_sexes_mask

        preferred_v = (active_preferences.unsqueeze(-1) * local_clan_v_avg).sum(dim=2)
        bias_weight = self.genes[..., Genes.FOLLOW_BIAS_VAL:Genes.FOLLOW_BIAS_VAL+1]
        tot_dv += preferred_v * bias_weight

        # Adjust for speed limits
        speed = torch.norm(tot_dv, dim=-1, keepdim=True).clamp(min=1e-5)
        too_fast = (speed > MAX_SPEED).float()
        too_slow = (speed < MIN_SPEED).float()

        # Apply the limits algebraically across the batch
        tot_dv = tot_dv * (1 - too_fast - too_slow) + \
                 (tot_dv / speed) * MAX_SPEED * too_fast + \
                 (tot_dv / speed) * MIN_SPEED * too_slow

        self.v = tot_dv

        return collisions