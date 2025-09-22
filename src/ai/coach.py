from config import coach_config
from src.ai.agent import Agent
import torch



class Coach:
    """Coach: suggests action or adjusts difficulty."""

    def __init__(self, mode=coach_config.DEFAULT_MODE, agent: Agent = None):
        self.mode = mode
        self.agent = agent

