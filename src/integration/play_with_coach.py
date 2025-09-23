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


    try:
        while True:
            action = 0
            for event in __import__('pygame').event.get():
                if event.type == __import__('pygame').QUIT:
                    game.close()
                    return
                elif event.type == __import__('pygame').KEYDOWN:
                    if event.key == __import__('pygame').K_SPACE:
                        action = 1

            # Step the game
            state, reward, done, info = game.step(action)

            # Coach behavior
            now = time.time()

