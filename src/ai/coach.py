from config import coach_config
from src.ai.agent import Agent
import torch



class Coach:
    """Coach: suggests action or adjusts difficulty."""

    def __init__(self, mode=coach_config.DEFAULT_MODE, agent: Agent = None):
        self.mode = mode
        self.agent = agent



    def suggest(self, state):
        # state = [y_norm, vel_norm, dist_norm, gap_top_norm]

        if self.agent is None:
            # simple heuristic: flap if bird below gap center
            bird_y = state[0]
            gap_top = state[3]
            gap_center = gap_top + 0.5 * (coach_config.DIFFICULTY_ADJUST_STEP / 100.0)  # slight heuristic
            
            if bird_y > gap_top + 0.05:
                return 1, 0.6
            return 0, 0.6
        
