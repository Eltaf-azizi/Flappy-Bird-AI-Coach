import pygame
from config import settings



class Bird:
    def __init__(self, x=settings.BIRD_X, y=settings.BIRD_START_Y):
        self.x = int(x)
        self.y = float(y)
        self.vel = 0.0
        self.width = 34
        self.height = 24
        self.rect = pygame.Rect(self.x, int(self.y), self.width, self.height)
        self.alive = True


    def flap(self):
        self.vel = settings.FLAP_VELOCITY