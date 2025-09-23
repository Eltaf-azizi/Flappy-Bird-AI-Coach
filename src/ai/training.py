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

    rewards_log = []
    start_time = time.time()


    for ep in trange(episodes, desc='Episodes'):
        state = env.reset()
        ep_reward = 0.0
        info = {}
        for step in range(settings.MAX_STEPS_PER_EPISODE):
            action = agent.act(state, eval_mode=False)
            next_state, reward, done, info = env.step(action)
            agent.push(state, action, reward, next_state, float(done))
            loss = agent.learn()
            state = next_state
            ep_reward += float(reward)
            total_steps += 1
            if total_steps % settings.TARGET_UPDATE_FREQ == 0:
                agent.update_target()
            if done:
                break
            