import random
from config import settings

def random_pipe_gap(gap_size=None):
    if gap_size is None:
        gap_size = settings.PIPE_GAP_SIZE
    min_y = 50
    max_y = settings.SCREEN_HEIGHT - gap_size - 50
    return random.randint(min_y, max_y)

def get_next_pipe(bird_x, pipes):
    for p in pipes:
        if p.x + p.width > bird_x:
            return p
    return None
