# Game & training settings
SCREEN_WIDTH = 288
SCREEN_HEIGHT = 512
FPS = 60

# Bird
BIRD_X = 50
BIRD_START_Y = SCREEN_HEIGHT // 2

# Pipes
PIPE_GAP_SIZE = 100
PIPE_WIDTH = 52
PIPE_SPEED = 3
PIPE_SPAWN_INTERVAL = 90  # frames

# Physics
GRAVITY = 0.5
FLAP_VELOCITY = -7.5
MAX_DROP_SPEED = 10

# RL
STATE_SIZE = 4  # [bird_y_norm, bird_vel_norm, dist_norm, gap_top_norm]
ACTION_SIZE = 2  # 0: do nothing, 1: flap

# Training defaults
TRAIN_EPISODES = 2000
MAX_STEPS_PER_EPISODE = 10000
BATCH_SIZE = 64
GAMMA = 0.99
LR = 1e-3
BUFFER_SIZE = 20000
TARGET_UPDATE_FREQ = 1000  # steps (policy -> target)
