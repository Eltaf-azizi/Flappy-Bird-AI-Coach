import argparse
from src.ai.env import FlappyEnv
from src.ai.agent import Agent



def evaluate(model_path, episodes=5):
    env = FlappyEnv(render=True)
    agent = Agent()
    agent.load(model_path)
