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

        self.net = Neural(input_dim,output_dim).half()
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

    def save(self,save_dir):
        save_path = save_dir / f"mario_net_best.pt"
        torch.save(
            dict(
                model=self.net.state_dict(),
                exploration_rate=self.exploration_rate
            ),
            save_path
        )
        print(f"MarioNet saved to {save_path}")

    def load(self, load_path):
        if not load_path.exists():
            raise ValueError(f"{load_path} does not exist")

        ckp = torch.load(load_path, map_location=('cuda' if self.use_cuda else 'cpu'))
        exploration_rate = ckp.get('exploration_rate')
        state_dict = ckp.get('model')

        print(f"Loading model at {load_path} with exploration rate {exploration_rate}")
        self.net.load_state_dict(state_dict)
        self.exploration_rate = exploration_rate