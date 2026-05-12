import torch

from src.dna import NUM_SEXES, Genes
from src.phys_globals import *
from src.swarm import DroneSwarm

class DroneSwarmCPU(DroneSwarm):
    def __init__(self, genes, sexes, offset=torch.tensor([0,0,0])):
        super().__init__(genes, sexes, (genes.size(0),), offset)
        self.n = genes.size(0)

    def update(self, other_s, killed, *obstacles):
        collisions = self.boid(other_s, *obstacles)
        self.s += self.v * DELTA_T

        if killed is None:
            killed = torch.zeros_like(collisions)

        dead = (self.s[:, Z] <= 0).logical_or(collisions).logical_or(killed)

        collision_coords = self.s[collisions]

        # Remove dead drones from the simulation
        self.s = self.s[~dead]
        self.v = self.v[~dead]
        self.sexes = self.sexes[~dead]
        self.genes = self.genes[~dead]
        self.id = self.id[~dead]
        self.alive = self.alive[~dead] # To be reverse compatable w GPU version
        self.n = self.s.size(0)

        return self.n == 0, collision_coords