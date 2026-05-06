from enum import IntEnum

import torch

NUM_CLANS = 5
DELTA_T = 0.002

X=0; Y=1; Z=2
CEILING = 5.
TURN_FACTOR = 1.5
TURN_MARGIN = 0.1

COMM_RANGE = 0.15 # Arbitrary
MAX_SPEED = 6
MIN_SPEED = 2
COLLISION_DIST = 0.01

class Params(IntEnum):
    ALPHA=0     # Crowding
    BETA=1      # Alignment
    GAMMA=2     # Cohesion
    TURN=3      # Avoid boundaries
    PROTECTED=4 # How close they can get
    MARGIN=5    # How low they dare to swoop
    FEAR=6      # How much they avoid predators
    DESIRE=7    # How much they desire targets
    FOLLOW_BIAS_VAL=8   # How much each dron biases clan pull
    LISTEN_BIAS_VAL=9   # How much to weight recon from others
    FOLLOW_BIAS=10       # Preference to follow each clan
    LISTEN_BIAS=10+NUM_CLANS    # Preference to listen to each clan

class DroneFlock:
    def __init__(self, n, clans, offset=torch.tensor([0,0,0]), genes=None):
        self.n = n

        self.s = torch.stack([
            self.__starting_loc(offset)
            for _ in range(n)
        ])
        self.v = (torch.rand(n, 3) - 0.5) * 2
        self.alive = torch.ones(n, dtype=torch.bool)

        self.clans = torch.zeros((n,NUM_CLANS))
        self.clans[torch.arange(n), clans] = 1

        if genes is None:
            # Decent starting params after some fiddling
            self.genes = torch.tensor([[
                0.05, 0.05, 0.0005,
                0.2, 0.1, 0.1,
                0, 0.25, 0.05, 0.1
            ]]).repeat(n, 1)

            # Initialize clan biases randomly
            b1 = 1-2*torch.rand(n, NUM_CLANS)
            b2 = 1-2*torch.rand(n, NUM_CLANS)

            self.genes = torch.cat([self.genes, b1, b2], dim=1)

        else:
            self.genes = genes

        # Genes will go away when agents die. This keeps a backup copy
        self.genepool = self.genes
        self.id = torch.arange(self.n)

    def __starting_loc(self, offset=torch.zeros(3)):
        s = (torch.rand(3) - 0.5) * 0.5
        s[Z] = 1
        s += offset
        return s

    def update(self, other_s, killed=None):
        collisions = self.boid(other_s)
        self.s += self.v * DELTA_T

        if killed is None:
            killed = torch.zeros_like(collisions)

        dead = (self.s[:, Z] <= 0).logical_or(collisions).logical_or(killed)

        collision_coords = self.s[collisions]

        # Remove dead drones from the simulation
        self.s = self.s[~dead]
        self.v = self.v[~dead]
        self.clans = self.clans[~dead]
        self.genes = self.genes[~dead]
        self.id = self.id[~dead]
        self.n = self.s.size(0)

        return self.n == 0, collision_coords

    def boid(self, other_s):
        # Could use F.pdist but don't want to deal with indexing
        # the upper triangle rn. TODO use that, bc it's a little faster
        dist = torch.norm(self.s[:, None]-self.s, dim=2, p=2)
        collisions = (dist.fill_diagonal_(float('inf')) < COLLISION_DIST).sum(dim=1).bool()

        # Step 1: Uncrowd the boids
        too_close = dist < self.genes[:, Params.PROTECTED:Params.PROTECTED+1]
        too_close = too_close.fill_diagonal_(0)
        close_count = too_close.sum(dim=1, keepdim=True)

        dv = too_close.float() @ self.s # Add positions of birds that are too close
        close_dv = (self.s*close_count - dv) * self.genes[:, Params.ALPHA:Params.ALPHA+1] # And move the boids away from them

        # Step 1.5: Find neighbors
        visible = dist < COMM_RANGE
        visible.fill_diagonal_(0) # Don't align with yourself
        neighbors = visible.sum(dim=1, keepdim=True)
        has_neighbors = (neighbors > 0).float()

        # Step 2: Align the boids
        dv = visible.float() @ self.v
        dv /= neighbors.clamp(min=1)
        align_dv = (dv - self.v) * has_neighbors * self.genes[:, Params.BETA:Params.BETA+1]

        # Step 3: Cohesion
        dv = visible.float() @ self.s
        dv /= neighbors.clamp(min=1)
        cohesion_dv = (dv - self.s) * has_neighbors * self.genes[:, Params.GAMMA:Params.GAMMA+1]

        # Step 4: Boundary avoidance
        min_bounds = self.s.new_tensor([TURN_MARGIN, TURN_MARGIN, 0]).repeat(self.n, 1)
        min_bounds[:, Z] = self.genes[:, Params.MARGIN]
        max_bounds = self.s.new_tensor([1.0, 1.0, CEILING]) - TURN_MARGIN

        # Calculate penetration depth into the margins
        too_low = (min_bounds - self.s).clamp(min=0)
        too_high = (self.s - max_bounds).clamp(min=0)

        # Apply standard TURN factor to all axes (Walls and Ceiling)
        turn_dv = (too_low - too_high) * TURN_FACTOR

        # Override ground avoidance
        ground_push = too_low[:, Z] * self.genes[:, Params.TURN]
        ceiling_push = too_high[:, Z] * TURN_FACTOR
        turn_dv[:, Z] = ground_push - ceiling_push

        tot_dv = self.v + close_dv + align_dv + cohesion_dv + turn_dv

        # Step 5: Enemy Interaction
        if other_s is not None and other_s.numel() > 0:
            # Calculate distances to all enemies -> Shape: (N_self, M_other)
            enemy_dist = torch.norm(self.s[:, None] - other_s, dim=2, p=2)
            enemy_collisions = (enemy_dist < COLLISION_DIST).sum(dim=1).bool()
            collisions = collisions.logical_or(enemy_collisions)

            # Find enemies within vision
            enemy_visible = enemy_dist < COMM_RANGE
            enemy_counts = enemy_visible.sum(dim=1, keepdim=True)
            has_enemies = (enemy_counts > 0).float()

            # Calculate the center of mass of visible enemies
            # (N, M) @ (M, 3) -> (N, 3)
            enemy_sum = enemy_visible.float() @ other_s
            enemy_center = enemy_sum / enemy_counts.clamp(min=1)

            # Calculate forces
            desire_dv = (enemy_center - self.s) * has_enemies * self.genes[:, Params.DESIRE:Params.DESIRE+1]
            fear_dv = (self.s - enemy_center) * has_enemies * self.genes[:, Params.FEAR:Params.FEAR+1]

            # Apply to total velocity
            tot_dv += desire_dv + fear_dv

            # Step 6: Send intel to others
            valid_senders = self.clans * has_enemies
            broadcast = valid_senders.unsqueeze(-1) * enemy_center.unsqueeze(1) # N x clans x 3

            # Assume drones can communicate with all others
            global_msg_sum = broadcast.sum(dim=0)           # Shape: (C, 3)
            global_msg_counts = valid_senders.sum(dim=0)    # Shape: (C,)
            msg = global_msg_sum / global_msg_counts.clamp(min=1).unsqueeze(-1)

            # Respond to received messages
            listen_prefs = self.genes[:, Params.LISTEN_BIAS : Params.LISTEN_BIAS + NUM_CLANS]
            has_signal = (global_msg_counts > 0).float().unsqueeze(0)
            active_listen_prefs = listen_prefs * has_signal

            # CALCULATE DIRECTION FIRST: Vector to each reported enemy center (N, C, 3)
            channel_dv = msg.unsqueeze(0) - self.s.unsqueeze(1)

            # Multiply by preferences. Negative prefs will naturally push them away!
            radio_dv = (active_listen_prefs.unsqueeze(-1) * channel_dv).sum(dim=1)
            radio_dv *= self.genes[:, Params.LISTEN_BIAS_VAL : Params.LISTEN_BIAS_VAL + 1]

            tot_dv += radio_dv

        # Step 7: Follow biased members
        local_clan_counts = visible.float() @ self.clans

        # Map velocities into their clan channels: (N, C, 3)
        clan_v_split = self.clans.unsqueeze(-1) * self.v.unsqueeze(1)
        local_clan_v_sum = visible.float() @ clan_v_split.view(self.n, NUM_CLANS * 3)
        local_clan_v_sum = local_clan_v_sum.view(self.n, NUM_CLANS, 3)

        local_clan_v_avg = local_clan_v_sum / local_clan_counts.clamp(min=1).unsqueeze(-1)

        preferences = self.genes[:, Params.FOLLOW_BIAS:Params.FOLLOW_BIAS + NUM_CLANS]
        visible_clans_mask = (local_clan_counts > 0).float()
        active_preferences = preferences * visible_clans_mask

        # Sum the weighted velocities directly. Negative preferences will cause
        # the drone to want to fly in the exact opposite direction!
        preferred_v = (active_preferences.unsqueeze(-1) * local_clan_v_avg).sum(dim=1)

        # Apply the weight as a standard additive force (replacing lerp)
        bias_weight = self.genes[:, Params.FOLLOW_BIAS_VAL:Params.FOLLOW_BIAS_VAL+1]
        tot_dv += preferred_v * bias_weight

        # Adjust for speed limits
        speed = torch.sqrt(tot_dv.pow(2).sum(dim=1))
        too_fast = speed > MAX_SPEED
        too_slow = speed < MIN_SPEED
        tot_dv[too_fast] = (tot_dv[too_fast]/speed[too_fast, None]) * MAX_SPEED
        tot_dv[too_slow] = (tot_dv[too_slow]/speed[too_slow, None]) * MIN_SPEED

        self.v = tot_dv

        return collisions