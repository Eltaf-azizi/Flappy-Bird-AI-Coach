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
        
        else:
            self.agent.policy_net.eval()
            with torch.no_grad():
                s = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                q = self.agent.policy_net(s)
                probs = torch.nn.functional.softmax(q, dim=1).cpu().numpy()[0]
                action = int(probs.argmax())
                confidence = float(probs.max())
                return action, confidence



    def adjust_difficulty(self, game):
        score = game.world.score
        # naive rule: if low score, increase gap_size (easier). if high score, reduce gap_size (harder).
        if score < 3:
            game.gap_size = min(game.gap_size + coach_config.DIFFICULTY_ADJUST_STEP, coach_config.MAX_GAP)
        else:
            game.gap_size = max(game.gap_size - coach_config.DIFFICULTY_ADJUST_STEP, coach_config.MIN_GAP)
