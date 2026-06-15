import numpy as np
import matplotlib.pyplot as plt
from env import ProbeEnv
from agent import ProbeAgent

def train_agent(episodes):
    env = ProbeEnv()
    agent = ProbeAgent()
    rewards = []
    results = []
    for epoch in range(episodes):
        current_state = agent.discretize_state(env.reset())
        total_reward = 0
        done = False
        while not done:
            action = agent.choose_action(current_state)
            next_state, reward, done, info = env.step(action)
            next_state = agent.discretize_state(next_state)
            agent.learn(current_state, action, reward, next_state, done)
            current_state = next_state
            total_reward += reward
        rewards.append(total_reward)
        results.append('Timeout' if info['timeout'] else 'Crash' if info['crash'] else 'Success')
        agent.epsilon *= agent.epsilon_decay
    print(results.count("Success"))
    print(results.count("Crash"))
    print(results.count("Timeout"))
    return rewards, results, agent

if __name__ == "__main__":
    episodes = 20000
    rewards, results, agent = train_agent(episodes)
    np.save('q_table.npy', agent.q_table)
    moving_rewards = []
    moving_results = []

    window = 100
    for i in range(episodes):
        start = max(0, i - window + 1)
        moving_rewards.append(np.mean(rewards[start:i+1]))
        moving_results.append(np.mean([1 if r == 'Success' else 0 for r in results[start:i+1]]))
    fig, ax = plt.subplots(2,1, figsize=(10,8))
    ax[0].plot(moving_rewards)
    ax[1].plot(moving_results)
    ax[0].set_title('Moving Average Reward')
    ax[1].set_title('Moving Average Success Rate')
    plt.tight_layout()
    plt.savefig('learning_curve.png')
    plt.show()