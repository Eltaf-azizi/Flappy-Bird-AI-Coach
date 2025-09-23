import argparse
from src.ai.env import FlappyEnv
from src.ai.agent import Agent



def evaluate(model_path, episodes=5):
    env = FlappyEnv(render=True)
    agent = Agent()
    agent.load(model_path)
    for ep in range(episodes):
        
        state = env.reset()
        done = False
        info = {}
        while not done:
            action = agent.act(state, eval_mode=True)
            state, reward, done, info = env.step(action)
            env.render()
        print('Episode', ep, 'score', info.get('score'))
    env.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--episodes', type=int, default=5)
    args = parser.parse_args()
    evaluate(args.model, episodes=args.episodes)

