import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from config import settings
import os

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class DQN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim)
        )

    
    def forward(self, x):
        return self.net(x)



class ReplayBuffer:
    def __init__(self, capacity=settings.BUFFER_SIZE):
        self.buffer = deque(maxlen=int(capacity))


    def push(self, state, action, reward, next_state, done):
        self.buffer.append((np.array(state, dtype=np.float32),
                            int(action),
                            float(reward),
                            np.array(next_state, dtype=np.float32),
                            float(done)))
        
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, ns, d = map(np.array, zip(*batch))
        return s, a, r, ns, d


    def __len__(self):
        return len(self.buffer)



class Agent:
    def __init__(self, state_dim=settings.STATE_SIZE, action_dim=settings.ACTION_SIZE, lr=settings.LR):
        self.policy_net = DQN(state_dim, action_dim).to(DEVICE)
        self.target_net = DQN(state_dim, action_dim).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.replay = ReplayBuffer()
        self.steps = 0
        self.eps_start = 1.0
        self.eps_end = 0.01
        self.eps_decay = 20000  # steps to decay
        self.gamma = settings.GAMMA
