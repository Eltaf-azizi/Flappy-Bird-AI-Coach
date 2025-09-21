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
    


    def step(self, action):
        # action: 0 nothing, 1 flap
        if int(action) == 1:
            self.bird.flap()
        self.bird.update()
        # spawn pipes
        if self.spawn_timer <= 0:
            gap_y = random_pipe_gap(self.gap_size)
            self.world.spawn_pipe(settings.SCREEN_WIDTH + 10, gap_y, gap_size=self.gap_size)
            self.spawn_timer = self.spawn_interval
        else:
            self.spawn_timer -= 1
        self.world.update()

        # collisions and rewards
        done = False
        reward = 0.0
        # ground or ceiling collision
        if self.bird.y <= 0 or (self.bird.y + self.bird.height) >= settings.SCREEN_HEIGHT:
            done = True
            reward = -1.0
            self.bird.alive = False

        # pipe collisions and score
        for p in self.world.pipes:
            if p.collides_with(self.bird.rect):
                done = True
                reward = -1.0
                self.bird.alive = False
                break
            if not p.passed and (p.x + p.width) < self.bird.x:
                p.passed = True
                self.world.score += 1
                reward = 1.0

        # living penalty to encourage progress
        if not done and reward == 0.0:
            reward = -0.01

        state = self.get_state()
        info = {"score": self.world.score}
        return state, reward, done, info
    


    def get_state(self):
        next_pipe = get_next_pipe(self.bird.x, self.world.pipes)
        if next_pipe is None:
            dist = settings.SCREEN_WIDTH
            top_y = settings.SCREEN_HEIGHT // 2
        else:
            dist = next_pipe.x - self.bird.x
            top_y = next_pipe.gap_y
        # normalized state
        return [
            self.bird.y / settings.SCREEN_HEIGHT,
            (self.bird.vel + abs(settings.FLAP_VELOCITY)) / (abs(settings.FLAP_VELOCITY) + settings.MAX_DROP_SPEED),
            dist / settings.SCREEN_WIDTH,
            top_y / settings.SCREEN_HEIGHT
        ]
    


    def render_frame(self):
        if not self.render:
            return
        self.screen.fill((135, 206, 235))  # sky
        # pipes
        for p in self.world.pipes:
            pygame.draw.rect(self.screen, (34, 139, 34), p.top_rect)
            pygame.draw.rect(self.screen, (34, 139, 34), p.bottom_rect)
        # bird
        pygame.draw.rect(self.screen, (255, 223, 0), self.bird.rect)
        # score
        score_surf = self.font.render(f"Score: {self.world.score}", True, (0, 0, 0))
        self.screen.blit(score_surf, (10, 10))
        pygame.display.flip()
        self.clock.tick(settings.FPS)



    def close(self):
        pygame.quit()



if __name__ == '__main__':
    game = FlappyGame(render=True)
    state = game.reset()
    while True:
        action = 0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game.close()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    action = 1
        state, reward, done, info = game.step(action)
        game.render_frame()
        if done:
            print("Died. Score:", info.get("score"))
            time.sleep(1.0)
            state = game.reset()
