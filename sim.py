import torch

from drone_team import DroneFlock, NUM_CLANS

class Env:
    def __init__(self, num_blue, num_red):
        self.blue = DroneFlock(
            num_blue,
            torch.randint(0, NUM_CLANS, (num_blue,)),
            offset=torch.tensor([0.25, 0.25, 0])
        )

        self.red = DroneFlock(
            num_red,
            torch.randint(0, NUM_CLANS, (num_red,)),
            offset=torch.tensor([0.75, 0.75, 0])
        )

        self.n_blue = num_blue
        self.n_red = num_red

    def update(self):
        self.blue.update()
        self.red.update()

    def coords(self):
        return (
            self.blue.s[self.blue.alive],
            self.red.s[self.red.alive]
        )