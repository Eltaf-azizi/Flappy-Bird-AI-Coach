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