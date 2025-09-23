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
            
        
        rewards_log.append(ep_reward)
        # save best by score
        score = info.get('score', 0)
        if score > best_score:
            best_score = score
            agent.save(save_path)
        # periodic checkpoint
        if ep % 25 == 0:
            ckpt_path = os.path.join(CHECKPOINT_DIR, f'ckpt_ep{ep}.pth')
            agent.save(ckpt_path)

        # save simple rewards log (append)
        with open(os.path.join(LOG_DIR, 'rewards.log'), 'a') as f:
            f.write(json.dumps({'episode': ep, 'reward': ep_reward, 'score': int(info.get('score', 0))}) + "\n")

    duration = time.time() - start_time
    print(f"Training finished. Time: {duration:.1f}s. Best score: {best_score}")
    env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=settings.TRAIN_EPISODES)
    parser.add_argument('--save', default=os.path.join(MODEL_DIR, 'best_model.pth'))
    args = parser.parse_args()
    train(episodes=args.episodes, save_path=args.save)