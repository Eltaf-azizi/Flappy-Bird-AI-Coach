import matplotlib.pyplot as plt
import json
import os


def plot_rewards_from_log(log_path='logs/rewards.log', out_path='logs/rewards.png'):
    if not os.path.exists(log_path):
        raise FileNotFoundError(log_path)
    eps = []
    rewards = []
    with open(log_path, 'r') as f:
        for line in f:
            obj = json.loads(line.strip())
            eps.append(obj.get('episode'))
            rewards.append(obj.get('reward'))
    plt.figure()
    plt.plot(eps, rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Rewards')
    plt.savefig(out_path)
    plt.close()
