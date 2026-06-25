import numpy as np
import matplotlib.pyplot as plt
from env import Environment
from agent import Agent
from evaluation import mean_wait_time, p90_wait_time, jain_fairness_index
import pickle
import copy

workloads = []
def train_agent(episodes):
    env = Environment()
    agent = Agent()
    rewards = []
    mean_wait_times = []
    p90_wait_times = []
    jfis = []
    for epoch in range(episodes):
        if (epoch%20==0):
            print(f'Episode {epoch} done')
        current_state = env.reset()
        workloads.append(copy.deepcopy(env.processes))
        total_reward = 0
        done = False
        while not done:
            action = agent.choose_action(current_state)
            next_state, reward, done = env.step(action)
            agent.learn(current_state, action, reward, next_state, done)
            current_state = next_state
            total_reward += reward
        rewards.append(total_reward)
        mean_wait_times.append(mean_wait_time(env.processes))
        p90_wait_times.append(p90_wait_time(env.processes))
        jfis.append(jain_fairness_index(env.processes))
        if epoch<=agent.training_episodes:
            agent.epsilon *= agent.epsilon_decay
        else:
            agent.epsilon = 0
    return rewards, mean_wait_times, p90_wait_times, jfis, agent

if __name__ == "__main__":
    episodes = 20000
    rewards, mean_wait_times, p90_wait_times, jfs, agent = train_agent(episodes)
    np.save('q_table.npy', agent.q_table)
    moving_rewards = []
    moving_mean_wait_times = []
    moving_p90_wait_times = []
    moving_jfis = []

    window = 100
    for i in range(episodes):
        start = max(0, i-window+1)
        moving_rewards.append(np.mean(rewards[start:i+1]))
        moving_mean_wait_times.append(np.mean(mean_wait_times[start:i+1]))
        moving_p90_wait_times.append(np.mean(p90_wait_times[start:i+1]))
        moving_jfis.append(np.mean(jfs[start:i+1]))
    
    fig, ax = plt.subplots(4, 1, figsize = (10,8))
    ax[0].plot(moving_rewards)
    ax[1].plot(moving_mean_wait_times)
    ax[2].plot(moving_p90_wait_times)
    ax[3].plot(moving_jfis)
    ax[0].set_title('Moving Average Reward')
    ax[1].set_title('Moving Mean Wait Time')
    ax[2].set_title('Moving p90 Wait Time')
    ax[3].set_title('Moving JFI')
    plt.tight_layout()
    plt.savefig('learning_curve.png')
    plt.show()

    data = {
        "eval_workloads": workloads,
        "moving_rewards": moving_rewards,
        "moving_mean_wait_times": moving_mean_wait_times,
        "moving_p90_wait_time": moving_p90_wait_times,
        "moving_jfs": moving_jfis
    }
    with open("saved_rl_agent_data.pkl", "wb") as f:
        pickle.dump(data, f)
