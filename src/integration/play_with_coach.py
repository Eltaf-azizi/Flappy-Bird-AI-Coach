"""Play the game interactively; coach prints suggestions to console."""

import argparse
import time
from src.game.flappy import FlappyGame
from src.ai.coach import Coach
from src.ai.agent import Agent



def main(mode='hint', model_path=None):
    game = FlappyGame(render=True)
    agent = None

    if model_path:
        agent = Agent()
        agent.load(model_path)
    coach = Coach(mode=mode, agent=agent)

    state = game.reset()
    last_hint = 0.0

