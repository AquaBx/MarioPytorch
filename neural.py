import copy

from torch import nn

class Neural(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.online = nn.Sequential(
            nn.Linear(input_dim,input_dim*output_dim*10),
            nn.Linear(input_dim*output_dim*10, output_dim),
            nn.Flatten(0, 1)
        )

        self.target = copy.deepcopy(self.online)

        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, input_data, model):
        if model == 'online':
            return self.online(input_data)
        elif model == 'target':
            return self.target(input_data)
