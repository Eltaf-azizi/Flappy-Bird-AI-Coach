import numpy as np
from src.game.flappy import FlappyGame
from config import settings


class FlappyEnv:
    """Gym-like wrapper around FlappyGame."""
    def __init__(self, render=False, gap_size=None):
        self.game = FlappyGame(render=render, gap_size=gap_size)
        self.observation_space = (settings.STATE_SIZE,)
        self.action_space = settings.ACTION_SIZE

    def reset(self):
        s = self.game.reset()
        return np.array(s, dtype=np.float32)

    def step(self, action):
        state, reward, done, info = self.game.step(int(action))
        return np.array(state, dtype=np.float32), float(reward), bool(done), info

    def render(self):
        self.game.render_frame()

    def close(self):
        self.game.close()
