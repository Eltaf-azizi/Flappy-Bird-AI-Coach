import os
import argparse
from tqdm import trange
from src.ai.env import FlappyEnv
from src.ai.agent import Agent
from config import settings
import json
import time

LOG_DIR = 'logs'
MODEL_DIR = 'models'
CHECKPOINT_DIR = os.path.join(MODEL_DIR, 'checkpoints')

def train(episodes=settings.TRAIN_EPISODES, save_path=os.path.join(MODEL_DIR, 'best_model.pth')):
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    env = FlappyEnv(render=False)
    agent = Agent()
    total_steps = 0
    best_score = -1