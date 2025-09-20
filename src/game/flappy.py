import pygame
import sys
import time
import random
from config import settings
from .objects import Bird, World
from .utils import random_pipe_gap, get_next_pipe

class FlappyGame:
    def __init__(self, render=True, gap_size=None):
        pygame.init()
        self.render = render
        self.screen = None
        if render:
            self.screen = pygame.display.set_mode((settings.SCREEN_WIDTH, settings.SCREEN_HEIGHT))
            pygame.display.set_caption('Flappy Bird - AI Coach')