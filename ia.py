import numpy as np
import torch
from neural import Neural

class IA:
    def __init__(self, input_dim : int, output_dim : int):

        self.total_reward = 0

        self.mutation_rate = 0.1

        self.current_step = 0
        self.exploration_rate_min = 0.1
        self.exploration_rate_decay = 0.99999975
        self.input_dim = input_dim
        self.exploration_rate = 1

        self.use_cuda = torch.cuda.is_available()

        self.net = Neural(input_dim,output_dim).float()
        if self.use_cuda:
            self.net = self.net.to(device='cuda')

    def act(self, state):
        if False and np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(self.input_dim)
        else :
            state = state.cuda() if self.use_cuda else state
            state = state.unsqueeze(0)
            actions_values = self.net(state, model = 'online')
            action_idx = torch.argmax(actions_values).item()

        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        self.current_step += 1
        return action_idx

    def mutate(self):
        for param in self.net.parameters():
            mutation_noise = torch.randn_like(param) * 0.5
            mask = torch.rand_like(param) < self.mutation_rate
            param.data += mask * mutation_noise

    def reward(self, score):
        self.total_reward += score

    def reset(self):
        self.total_reward = 0