from enum import IntEnum
import pickle

import torch

NUM_SEXES = 5
MAX_GAME_LEN = 2000

class Genes(IntEnum):
    ALPHA=0     # Crowding
    BETA=1      # Alignment
    GAMMA=2     # Cohesion
    TURN=3      # Avoid boundaries
    OBSTACLE_AVOID=4 # How daring they are around obstacles
    PROTECTED=5 # How close they can get to each other
    MARGIN=6    # How low they dare to swoop
    FEAR=7      # How much they avoid predators
    CAN_FIRE=8  # Make some drones only scouts
    COMM_RANGE=9        # How far messages propagate
    VIZ_RANGE=10         # How far they can see
    FOLLOW_BIAS_VAL=11   # How much each dron biases clan pull
    LISTEN_BIAS_VAL=12   # How much to weight recon from others
    FOLLOW_BIAS=13       # Preference to follow each clan
    LISTEN_BIAS=13+NUM_SEXES    # Preference to listen to each clan

    LEN = LISTEN_BIAS+NUM_SEXES

# Decent starting params after some fiddling
DEFAULT = torch.tensor([
    0.05, 0.05, 0.0005, # Flight params (0-2)
    0.2, 0.2, 0.1, 0.1, # Boundary avoidance (3-6)
    0.25,               # Fear (7)
    0.1,                # Can fire (8)
    0.25, 0.25,         # Comm/viz range (NORMALIZED!)
    0., 0.              # Biases (12-13)
])

NORMALIZE = [Genes.COMM_RANGE, Genes.VIZ_RANGE]
BONUS = 2

class GenePool:
    def __init__(self, population, xover_rate=0.75, mute_rate=0.05,
                 mute_stren=0.25, xover_alpha=0.1,
                 device='cpu', use_baseline=False, hybrid_init=False,
                 init_mutate=False, tournament_size=5):

        self.baseline = torch.cat([
            DEFAULT.repeat(1,NUM_SEXES,1),
            torch.zeros(1,NUM_SEXES,NUM_SEXES),
            torch.zeros(1,NUM_SEXES,NUM_SEXES)
        ], dim=-1).to(device)
        self.baseline_meta = torch.ones(1,NUM_SEXES, device=device)

        # Boid params
        if use_baseline:
            self.genes = self.baseline.repeat(population,1,1)

        # Use boid flying params with randomized combat params
        elif hybrid_init:
            self.genes = DEFAULT.repeat(population, NUM_SEXES, 1).to(device)
            self.genes[..., Genes.FEAR:] = torch.rand(population, NUM_SEXES, Genes.FOLLOW_BIAS-Genes.FEAR, device=device) - 0.5
            follow_bias = 0.1 * torch.rand(population, NUM_SEXES, NUM_SEXES, device=device) - 0.05
            listen_bias = 0.1 * torch.rand(population, NUM_SEXES, NUM_SEXES, device=device) - 0.05

            self.genes = torch.cat([self.genes, follow_bias, listen_bias], dim=-1)

        else:
            self.genes = torch.rand(population, NUM_SEXES, Genes.LEN, device=device) - 0.5

        # Prob of generating children of each sex
        self.meta_genes = torch.rand(population, NUM_SEXES, device=device)

        self.population = population
        self.device = device

        # Evolution params
        self.mute_rate = mute_rate
        self.mute_stren = mute_stren
        self.xover_rate = xover_rate
        self.xover_alpha = xover_alpha
        self.tournament_size = tournament_size

        # Updated in evolve loop
        self.scores = torch.zeros(population, device=device)

        # For baseline init during training
        if init_mutate:
            self.mutate(self.genes, self.meta_genes)

    def tournament(self):
        tournies = torch.randint(
            0,self.population,
            (self.population*2, self.tournament_size),
            device=self.device
        )

        tourney_fitness = self.scores[tournies]
        winner_indices = tourney_fitness.argmax(dim=-1)
        parent_indices = tournies[torch.arange(self.population*2, device=self.device), winner_indices]

        return parent_indices.chunk(2)

    def reproduce(self, *args):
        # Rescue top 5%
        n_elite = int(self.population * 0.05)
        elites = self.scores.argsort(descending=True)
        elites = elites[:n_elite]

        p1, p2 = self.tournament()
        children = self.ab_crossover(p1,p2)

        genes, meta = self.mutate(*children)
        genes[:n_elite] = self.genes[elites]
        meta[:n_elite] = self.meta_genes[elites]

        self.genes = genes
        self.meta_genes = meta

    def create_swarm(self, swarm_size, queens=None):
        if queens is None:
            queens = torch.arange(self.population, device=self.device)

        b = queens.size(0)
        is_baseline = (queens == -1).unsqueeze(-1)

        child_sex_logits = self.meta_genes[queens] # Shape: (B, NUM_SEXES)
        child_sex_logits = torch.where(is_baseline, self.baseline_meta, child_sex_logits)

        child_sex_probs = torch.softmax(child_sex_logits, dim=-1)
        childrens_sexes = torch.multinomial(child_sex_probs, swarm_size, replacement=True) # Shape: (B, swarm_size)

        # Retrieve the Queen's full genome -> Shape: (B, NUM_SEXES, G)
        queen_genes = self.genes[queens]
        queen_genes = torch.where(is_baseline.unsqueeze(-1), self.baseline, queen_genes)

        b_idx = torch.arange(b, device=self.device).unsqueeze(1)

        swarm_genes = queen_genes[b_idx, childrens_sexes] # Resulting Shape: (B, swarm_size, G)
        swarm_genes[..., NORMALIZE] = torch.sigmoid(swarm_genes[..., NORMALIZE])

        # Tradeoffs
        # Drones without weapons have better intel
        scouts = swarm_genes[..., Genes.CAN_FIRE] < 0
        swarm_genes[..., Genes.VIZ_RANGE][scouts] *= BONUS
        swarm_genes[..., Genes.COMM_RANGE][scouts] *= BONUS

        return swarm_genes, childrens_sexes

    def save(self, outf):
        with open(outf, 'wb+') as f:
            pickle.dump(self, f)

    def diversity(self):
        '''
        Calculate avg distance from mean of each DNA sequence
        '''
        flat_dna = self.genes.view(self.population, -1)
        diversity = torch.pdist(flat_dna).mean().item()
        return diversity

    @staticmethod
    def load(fname, device=None):
        with open(fname, 'rb') as f:
            obj: GenePool = CPU_Unpickler(f).load()

        if device is not None:
            obj.genes = obj.genes.to(device)
            obj.meta_genes = obj.meta_genes.to(device)
            obj.baseline = obj.baseline.to(device)
            obj.baseline_meta = obj.baseline_meta.to(device)
            obj.scores = obj.scores.to(device)
            obj.device = device

        return obj


    def mutate(self, genes, meta):
        # Generate random noise between [-mute_stren, mute_stren]
        mute_noise_genes = (torch.rand_like(genes) * 2 - 1) * self.mute_stren
        mute_noise_meta = (torch.rand_like(meta) * 2 - 1) * self.mute_stren

        # Create boolean masks of where mutations should occur (converted to 1.0 or 0.0)
        to_mute_genes = (torch.rand_like(genes) < self.mute_rate).float()
        to_mute_meta = (torch.rand_like(meta) < self.mute_rate).float()

        # Add the noise only where the mask is 1.0.
        genes += mute_noise_genes * to_mute_genes
        meta += mute_noise_meta * to_mute_meta

        return genes, meta

    def ab_crossover(self, grp1, grp2):
        new_genes = self._ab_crossover(
            self.genes[grp1], self.genes[grp2]
        )

        new_meta_genes = self._ab_crossover(
            self.meta_genes[grp1], self.meta_genes[grp2]
        )

        return new_genes, new_meta_genes

    def _ab_crossover(self, p1, p2):
        '''
        Given two groups, perform BLX-\alpha\beta where genes
        related to the parents' sex are weighted with the alpha parameter
        genes related to neither parent's sex are weighted with the beta parameter
        '''
        # Find min and max boundaries
        g_min = torch.minimum(p1, p2)
        g_max = torch.maximum(p1, p2)
        delta = g_max - g_min

        # Calculate absolute bounds for the new gene
        lower_bound = g_min - (self.xover_alpha * delta)
        upper_bound = g_max + (self.xover_alpha * delta)

        # Generate two potential children from the distribution
        u1 = torch.rand_like(lower_bound)
        u2 = torch.rand_like(lower_bound)
        child1_genes = lower_bound + (upper_bound - lower_bound) * u1
        child2_genes = lower_bound + (upper_bound - lower_bound) * u2

        # Apply the crossover rate mask (if false, just keep the parent's gene)
        to_cross = torch.rand_like(p1) < self.xover_rate
        child1 = torch.where(to_cross, child1_genes, p1)
        child2 = torch.where(to_cross, child2_genes, p2)

        # Randomly select between the two offspring configurations (could be 2 or 3 dims)
        flip_shape = [child1.shape[0]] + [1] * (child1.dim() - 1)
        coin_flip = torch.rand(flip_shape, device=self.device) < 0.5
        children = torch.where(coin_flip, child1, child2)

        return children

    @staticmethod
    def fitness(bkills, rkills, b_won, r_won, game_len):
        '''
        Takes normalized count of kills (min 0, max 1)
        and normalized game len (min 0, max 1) and calculates
        the fitness of the queen that produced the swarm by

            kills + (1-game_len) * 1(if team won)

        '''
        bscore = bkills
        rscore = rkills

        if b_won:
            bscore += (1-game_len)
        elif r_won:
            rscore += (1-game_len)

        return bscore, rscore


import io
class CPU_Unpickler(pickle.Unpickler):
    '''
    Because pickle.dump with GPU 3 doesn't want to work on my
    local 1 GPU machine
    '''
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)