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


    
    def act(self, state, eval_mode=False):
        # epsilon-greedy
        eps = self.epsilon()
        if eval_mode:
            with torch.no_grad():
                s = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                q = self.policy_net(s)
                return int(torch.argmax(q, dim=1).item())
        # training mode
        if random.random() > eps:
            with torch.no_grad():
                s = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                q = self.policy_net(s)
                action = int(torch.argmax(q, dim=1).item())
        else:
            action = random.randrange(settings.ACTION_SIZE)
        self.steps += 1
        return action
    


    def epsilon(self):
        # linear decay
        return max(self.eps_end, self.eps_start - (self.eps_start - self.eps_end) * (self.steps / max(1, self.eps_decay)))



    def push(self, state, action, reward, next_state, done):
        self.replay.push(state, action, reward, next_state, done)



    def learn(self, batch_size=settings.BATCH_SIZE):
        if len(self.replay) < batch_size:
            return None
        
        s, a, r, ns, d = self.replay.sample(batch_size)
        s = torch.tensor(s, dtype=torch.float32, device=DEVICE)
        a = torch.tensor(a, dtype=torch.int64, device=DEVICE).unsqueeze(1)
        r = torch.tensor(r, dtype=torch.float32, device=DEVICE).unsqueeze(1)
        ns = torch.tensor(ns, dtype=torch.float32, device=DEVICE)
        d = torch.tensor(d, dtype=torch.float32, device=DEVICE).unsqueeze(1)

        q_vals = self.policy_net(s).gather(1, a)
        with torch.no_grad():
            q_next = self.target_net(ns).max(1)[0].unsqueeze(1)
            q_target = r + (1.0 - d) * self.gamma * q_next
        loss = nn.functional.mse_loss(q_vals, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())



    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())



    def save(self, path):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        torch.save(self.policy_net.state_dict(), path)



    def load(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        self.policy_net.load_state_dict(torch.load(path, map_location=DEVICE))
        self.target_net.load_state_dict(self.policy_net.state_dict())

   