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
        self.clock = pygame.time.Clock()
        self.bird = Bird()
        self.world = World()
        self.gap_size = gap_size or settings.PIPE_GAP_SIZE
        self.running = True
        self.spawn_timer = 0
        self.spawn_interval = int(settings.PIPE_SPAWN_INTERVAL)
        self.font = pygame.font.SysFont('Arial', 16) if render else None



    def reset(self):
        self.bird = Bird()
        self.world = World()
        self.spawn_timer = 0
        self.running = True
        self.bird.alive = True
        return self.get_state()