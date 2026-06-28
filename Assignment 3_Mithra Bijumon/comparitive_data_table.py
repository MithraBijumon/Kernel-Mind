import numpy as np
from agent import Agent
from workload_generator import workload_generator
import pickle
import copy
from env import Environment
from evaluation import mean_wait_time, p90_wait_time, jain_fairness_index

with open('saved_rl_agent_data.pkl', 'rb') as f:
    rl_data = pickle.load(f)
workloads = rl_data["eval_workloads"]

with open('saved_baseline_data.pkl', 'rb') as f:
    baseline_data = pickle.load(f)
fcfs_avg = baseline_data["fcfs_avg"]
rr_avg = baseline_data["rr_avg"]
ra_avg = baseline_data["ra_avg"]
sjf_avg = baseline_data["sjf_avg"]

agent = Agent()
agent.q_table = np.load("q_table.npy")
agent.epsilon = 0

rl_results = np.zeros((200,3))

for episode in range(200):
    if (episode % 20 == 0):
        print(f'Episode {episode} done')
    processes = copy.deepcopy(workloads[episode * 100])
    env = Environment()
    state = env.discretize_state(env.get_state())
    done = False
    while not done:
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        state = next_state
    result = [mean_wait_time(env.processes), p90_wait_time(env.processes), jain_fairness_index(env.processes)]
    rl_results[episode] = result

rl_avg = np.mean(rl_results, axis = 0)

print("---------------------------------------------------------------")
print("| Scheduler   | Mean Wait Time | P90 Wait Time | Jain Index   |")
print("---------------------------------------------------------------")
print(f"| FCFS        | {fcfs_avg[0]:12.2f} | {fcfs_avg[1]:13.2f} | {fcfs_avg[2]:11.4f} |")
print(f"| Round Robin | {rr_avg[0]:12.2f} | {rr_avg[1]:13.2f} | {rr_avg[2]:11.4f} |")
print(f"| SJF         | {sjf_avg[0]:12.2f} | {sjf_avg[1]:13.2f} | {sjf_avg[2]:11.4f} |")
print(f"| Random      | {ra_avg[0]:12.2f} | {ra_avg[1]:13.2f} | {ra_avg[2]:11.4f} |")
print(f"| RL Agent    | {rl_avg[0]:12.2f} | {rl_avg[1]:13.2f} | {rl_avg[2]:11.4f} |")
print("---------------------------------------------------------------")