"""Simple entrypoint to start training."""

from src.ai.training import train
from config import settings
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=settings.TRAIN_EPISODES)
    parser.add_argument('--save', default='models/best_model.pth')
    args = parser.parse_args()
    train(episodes=args.episodes, save_path=args.save)

