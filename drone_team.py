import torch

NUM_ROLES = 10
DELTA_T = 0.01
FRICTION = 0.95

X=0; Y=1; Z=2

class DroneFlock:
    def __init__(self, n, roles, offset=torch.tensor([0,0,0])):
        self.s = torch.stack([
            self.__starting_loc(offset)
            for _ in range(n)
        ])
        self.v = torch.zeros(n, 3)
        self.a = (torch.rand(n, 3) - 0.5) * 2 * DELTA_T

        self.roles = torch.zeros((n,NUM_ROLES))
        self.roles[torch.arange(n), roles] = 1
        self.alive = torch.ones(n, dtype=torch.bool)

    def __starting_loc(self, offset=torch.zeros(3)):
        s = torch.rand(3) * 0.01
        s[Z] = 1
        s += offset

        return s

    def update(self):
        self.v += (self.a * DELTA_T) * FRICTION
        self.s += self.v

        dead = self.s[:, Z] <= 0
        killed = torch.logical_and(dead, ~self.alive)

        self.alive[dead] = False
        self.a[dead] = torch.zeros(3)
        self.v[dead] = torch.zeros(3)

        return killed

    def move(self, vec):
        self.a += vec

    def message(self):
        return torch.cat([
            self.v,
            self.a,
            self.s,
            self.roles
        ])