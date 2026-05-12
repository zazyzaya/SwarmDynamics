import pandas as pd
import torch

from src.env import Env
from src.cpu.swarm import DroneSwarmCPU

class EnvCPU(Env):
    def __init__(self, blue_genes, blue_sexes, red_genes, red_sexes, obstacles=None):
        device = blue_genes.device

        blue_swarm = DroneSwarmCPU(blue_genes, blue_sexes, offset=torch.tensor([0.25, 0.25, 0]).to(device))
        red_swarm = DroneSwarmCPU(red_genes, red_sexes, offset=torch.tensor([0.75, 0.75, 0]).to(device))

        self.n_blue = blue_genes.size(0)
        self.n_red = red_genes.size(0)

        if obstacles is None:
            obstacles = (torch.empty((0,2)), torch.empty((0, 1)), torch.empty((0,1)))
        self.obstacles = obstacles

        super().__init__(blue_swarm, red_swarm, (self.n_blue,), (self.n_red,), device, obstacles)

    def update(self, viz=False):
        b_killed, r_killed, b_new_kills, r_new_kills = self.attack()
        b_obstacle, r_obstacle = self.crashes()

        b_obstacle_pos = self.blue.s[b_obstacle]
        r_obstacle_pos = self.red.s[r_obstacle]

        explosion_pos = torch.cat([self.blue.s[b_killed], self.red.s[r_killed]])

        self.b_alive_time[self.blue.id] += 1
        self.r_alive_time[self.red.id] += 1

        self.b_kills[self.blue.id] += b_new_kills
        self.r_kills[self.red.id] += r_new_kills

        red_s = self.red.s.clone()
        blue_s = self.blue.s.clone()

        blue_loss, b_collisions = self.blue.update(red_s, b_killed | b_obstacle, *self.obstacles)
        red_loss, r_collisions = self.red.update(blue_s, r_killed | r_obstacle, *self.obstacles)

        b_collisions = torch.cat([b_obstacle_pos, b_collisions])
        r_collisions = torch.cat([r_obstacle_pos, r_collisions])

        if viz:
            return torch.tensor([blue_loss, red_loss]), explosion_pos, b_collisions, r_collisions
        else:
            return torch.tensor([blue_loss, red_loss])

    def _to_df(self, idx, is_blue):
        kills = self.b_kills if is_blue else self.r_kills
        alive = self.b_alive_time if is_blue else self.r_alive_time

        return pd.DataFrame({
            'kills': kills[idx].cpu(),
            'lifespan': alive[idx].cpu()
        }, index=idx.cpu().tolist())

    def get_stats(self, top_k=None):
        if top_k is None:
            top_k = max(self.n_blue, self.n_red)

        b_survivors = self.b_alive_time.sort(descending=True).indices[:top_k]
        b_killers = self.b_kills.sort(descending=True).indices[:top_k]
        r_survivors = self.r_alive_time.sort(descending=True).indices[:top_k]
        r_killers = self.r_kills.sort(descending=True).indices[:top_k]

        return (
            self._to_df(b_survivors, True),
            self._to_df(b_killers, True),
            self._to_df(r_survivors, False),
            self._to_df(r_killers, False)
        )

