import torch

from src.dna import NUM_SEXES, Genes
from src.swarm import DroneSwarm
from src.phys_globals import *

class DroneSwarmCUDA(DroneSwarm):
    def __init__(self, genes, sexes, offset=torch.tensor([0,0,0])):
        self.B, self.N = genes.shape[:2]
        super().__init__(genes, sexes, genes.shape[:2], offset)


    def update(self, other_s, killed, obst_pos, obst_z):
        collisions = self.boid(other_s, obst_pos, obst_z)
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