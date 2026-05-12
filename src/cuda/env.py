import pandas as pd
import torch

from src.dna import Genes
from src.env import Env
from src.cuda.swarm import DroneSwarmCUDA
from src.phys_globals import RANGE, CYLINDER_RADIUS

class EnvCUDA(Env):
    def __init__(self, blue_genes, blue_sexes, red_genes, red_sexes, obstacles=None):
        device = blue_genes.device
        self.B, self.N = blue_genes.shape[:2]

        blue_swarm = DroneSwarmCUDA(blue_genes, blue_sexes, offset=torch.tensor([0.25, 0.25, 0]))
        red_swarm = DroneSwarmCUDA(red_genes, red_sexes, offset=torch.tensor([0.75, 0.75, 0]))

        if obstacles is None:
            obstacles = (torch.empty((self.B, 0, 3,2)), torch.empty((self.B, 0, 1)))
        self.obstacles = obstacles

        super().__init__(blue_swarm, red_swarm, (self.B, self.N), (self.B, self.N), device, obstacles)

    def update(self):
        b_killed, r_killed, b_new_kills, r_new_kills = self.attack()
        b_obstacle, r_obstacle = self.crashes()

        self.b_alive_time += self.blue.alive.float()
        self.r_alive_time += self.red.alive.float()
        self.b_kills += b_new_kills
        self.r_kills += r_new_kills

        red_s = self.red.s.clone()
        blue_s = self.blue.s.clone()

        blue_loss, _ = self.blue.update(red_s, b_killed | b_obstacle, *self.obstacles)
        red_loss, _ = self.red.update(blue_s, r_killed | r_obstacle, *self.obstacles)
        game_over = torch.stack([blue_loss, red_loss], dim=1)

        # B x 2
        return game_over
