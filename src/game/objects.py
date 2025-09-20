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


    def update(self):
        self.vel += settings.GRAVITY
        if self.vel > settings.MAX_DROP_SPEED:
            self.vel = settings.MAX_DROP_SPEED
        self.y += self.vel
        self.rect.topleft = (self.x, int(self.y))



class Pipe:
    def __init__(self, x, gap_y, gap_size=settings.PIPE_GAP_SIZE):
        self.x = float(x)
        self.gap_y = int(gap_y)
        self.gap_size = int(gap_size)
        self.width = int(settings.PIPE_WIDTH)
        self.top_rect = pygame.Rect(int(self.x), 0, self.width, int(self.gap_y))
        self.bottom_rect = pygame.Rect(int(self.x), int(self.gap_y + self.gap_size),
                                       self.width, settings.SCREEN_HEIGHT - int(self.gap_y + self.gap_size))
        self.passed = False