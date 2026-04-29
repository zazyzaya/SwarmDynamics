from random import randint

import torch

from drone_team import DroneFlock, NUM_ROLES

class Env:
    def __init__(self, num_blue, num_red):
        self.blue = DroneFlock(
            num_blue,
            torch.randint(0, NUM_ROLES, (num_blue,)),
            offset=torch.tensor([-0.25, -0.25, 0])
        )

        self.red = DroneFlock(
            num_red,
            torch.randint(0, NUM_ROLES, (num_red,)),
            offset=torch.tensor([0.25, 0.25, 0])
        )

        self.n_blue = num_blue
        self.n_red = num_red

    def update(self):
        dv = 2*(torch.rand(self.n_blue, 3) - 0.5) * 0.00001
        self.blue.move(dv)
        self.blue.update()

        dv = 2*(torch.rand(self.n_red, 3) - 0.5) * 0.00001
        self.red.move(dv)
        self.red.update()

    def coords(self):
        return (
            self.blue.s[self.blue.alive],
            self.red.s[self.red.alive]
        )