from enum import IntEnum
import pickle

import torch

NUM_SEXES = 5

class Genes(IntEnum):
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
    LISTEN_BIAS=10+NUM_SEXES    # Preference to listen to each clan

    LEN = LISTEN_BIAS+NUM_SEXES
    N_BIASES = 2

# Decent starting params after some fiddling
DEFAULT = torch.tensor([
    0.05, 0.05, 0.0005,
    0.2, 0.1, 0.1,
    0, 0.25, 0.05, 0.1
])

class GenePool:
    def __init__(self, population, xover_rate=0.75, mute_rate=0.01,
                 mute_stren=0.25, xover_alpha=0.1, xover_beta=0.05,
                 device='cpu', use_baseline=False, hybrid_init=False):
        # Boid params
        if use_baseline:
            self.genes = DEFAULT.repeat(population, NUM_SEXES, 1).to(device)
            follow_bias = torch.zeros(population, NUM_SEXES, NUM_SEXES, device=device)
            listen_bias = torch.zeros(population, NUM_SEXES, NUM_SEXES, device=device)

            self.genes = torch.cat([self.genes, follow_bias, listen_bias], dim=-1)

        # Use boid flying params with randomized combat params
        elif hybrid_init:
            self.genes = DEFAULT.repeat(population, NUM_SEXES, 1).to(device)
            self.genes[..., Genes.FEAR:] = torch.rand(population, NUM_SEXES, Genes.FOLLOW_BIAS-Genes.FEAR, device=device) - 0.5
            follow_bias = 0.1 * torch.rand(population, NUM_SEXES, NUM_SEXES, device=device) - 0.05
            listen_bias = 0.1 * torch.rand(population, NUM_SEXES, NUM_SEXES, device=device) - 0.05

            self.genes = torch.cat([self.genes, follow_bias, listen_bias], dim=-1)

        else:
            self.genes = torch.rand(population, NUM_SEXES, Genes.LEN, device=device) - 0.5

        # Prob of spawning children of sex `col` given current sex `row`
        self.meta_genes = torch.rand(population, NUM_SEXES, NUM_SEXES, device=device)
        self.sexes = torch.randint(0, NUM_SEXES, (population,), device=device)

        self.population = population
        self.device = device

        self.mute_rate = mute_rate
        self.mute_stren = mute_stren

        self.xover_rate = xover_rate
        self.xover_alpha = xover_alpha
        self.xover_beta = xover_beta

    def reproduce(self, winners):
        num_winners = winners.size(0)

        p1 = torch.randint(0, num_winners, (self.population,), device=self.device)
        p2 = torch.randint(0, num_winners, (self.population,), device=self.device)

        children = self.ab_crossover(
            winners[p1], winners[p2],
            self.sexes[p1], self.sexes[p2]
        )

        coinflip = torch.rand(p1.size(0), device=self.device) < 0.5
        self.sexes = self._select_sex(torch.where(coinflip, winners[p1], winners[p2]))

        genes, meta = self.mutate(*children)
        self.genes = genes
        self.meta_genes = meta

    def phenotype(self, organisms=None):
        if organisms is None:
            organisms = torch.arange(self.population)

        sexes = self.sexes[organisms]
        return self.genes[organisms, sexes], sexes

    def save(self, outf):
        with open(outf, 'wb+') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(fname):
        with open(fname, 'rb') as f:
            obj = pickle.load(f)

        return obj

    def _select_sex(self, dom_parent):
        parent_sex = self.sexes[dom_parent]
        prob = self.meta_genes[dom_parent]

        # Pull out the pdfs
        n = prob.size(0)
        child_sex_logits = prob[torch.arange(n), parent_sex]
        child_sex_probs = torch.softmax(child_sex_logits, dim=1)

        # Sample
        child_sex = torch.multinomial(child_sex_probs, 1)
        return child_sex.flatten()

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

    def ab_crossover(self, grp1, grp2, sex1, sex2):
        new_genes = self._ab_crossover(
            self.genes[grp1], self.genes[grp2],
            sex1, sex2
        )

        new_meta_genes = self._ab_crossover(
            self.meta_genes[grp1], self.meta_genes[grp2],
            sex1, sex2
        )

        return new_genes, new_meta_genes

    def _ab_crossover(self, p1, p2, sex1, sex2):
        '''
        Given two groups, perform BLX-\alpha\beta where genes
        related to the parents' sex are weighted with the alpha parameter
        genes related to neither parent's sex are weighted with the beta parameter
        '''
        # Bias traits related to parents' sex
        n = p1.size(0)

        coef_1 = torch.full_like(p1, self.xover_beta)
        coef_1[torch.arange(n), sex1, :] = self.xover_alpha
        coef_2 = torch.full_like(p2, self.xover_beta)
        coef_2[torch.arange(n), sex2, :] = self.xover_alpha

        # Find min and max boundaries
        g_min = torch.minimum(p1, p2)
        g_max = torch.maximum(p1, p2)
        delta = g_max - g_min

        # Apply the correct coefficient based on which parent was the minimum
        p1_is_min = p1 <= p2

        # If p1 is the lower bound, it expands using coef_1. Else, it expands using coef_2.
        coef_min = torch.where(p1_is_min, coef_1, coef_2)
        coef_max = torch.where(p1_is_min, coef_2, coef_1)

        # Calculate absolute bounds for the new gene
        lower_bound = g_min - (coef_min * delta)
        upper_bound = g_max + (coef_max * delta)

        # Generate two potential children from the distribution
        u1 = torch.rand_like(lower_bound)
        u2 = torch.rand_like(lower_bound)
        child1_genes = lower_bound + (upper_bound - lower_bound) * u1
        child2_genes = lower_bound + (upper_bound - lower_bound) * u2

        # Apply the crossover rate mask (if false, just keep the parent's gene)
        to_cross = torch.rand_like(p1) < self.xover_rate
        child1 = torch.where(to_cross, child1_genes, p1)
        child2 = torch.where(to_cross, child2_genes, p2)

        # Randomly select between the two offspring configurations
        coin_flip = torch.rand(child1.size(0), 1, 1) < 0.5
        children = torch.where(coin_flip, child1, child2)

        return children