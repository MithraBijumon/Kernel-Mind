import numpy as np
from fcfs import FCFS
from round_robin import round_robin
from random_agent import random_agent
from sjf import SJF
from workload_generator import workload_generator
from evaluation import mean_wait_time, p90_wait_time, jain_fairness_index
import copy
import pickle

with open ('saved_rl_agent_data.pkl', 'rb') as f:
    data = pickle.load(f)

workloads = data['eval_workloads']

no_of_episodes = 200

fcfs_results = np.zeros((no_of_episodes, 3))
rr_results = np.zeros((no_of_episodes, 3))
ra_results = np.zeros((no_of_episodes,3))
sjf_results = np.zeros((no_of_episodes,3))

for episode in range(no_of_episodes):
    processes = workloads[episode*100]

    fcfs_agent = FCFS(copy.deepcopy(processes))
    fcfs_agent.schedule()
    result = [mean_wait_time(fcfs_agent.processes), p90_wait_time(fcfs_agent.processes), jain_fairness_index(fcfs_agent.processes)]
    fcfs_results[episode] = (result)

    rr_agent = round_robin(copy.deepcopy(processes), 10)
    rr_agent.schedule()
    result = [mean_wait_time(rr_agent.processes), p90_wait_time(rr_agent.processes), jain_fairness_index(rr_agent.processes)]
    rr_results[episode] = (result)

    ra_agent = random_agent(copy.deepcopy(processes))
    ra_agent.schedule()
    result = [mean_wait_time(ra_agent.processes), p90_wait_time(ra_agent.processes), jain_fairness_index(ra_agent.processes)]
    ra_results[episode] = (result)

    sjf_agent = SJF(copy.deepcopy(processes))
    sjf_agent.schedule()
    result = [mean_wait_time(sjf_agent.processes), p90_wait_time(sjf_agent.processes), jain_fairness_index(sjf_agent.processes)]
    sjf_results[episode] = (result)
    if (episode % 20 == 0):
        print(f'{episode+1} episodes done')

fcfs_avg = np.mean(fcfs_results, axis=0)
rr_avg   = np.mean(rr_results, axis=0)
sjf_avg  = np.mean(sjf_results, axis=0)
ra_avg   = np.mean(ra_results, axis=0)

print("Mean Wait Time: ")
print(f'FCFS {fcfs_avg[0]} | RR {rr_avg[0]} | SJF {sjf_avg[0]} | RandomAgent {ra_avg[0]}')
print()

print("P90 Wait Time: ")
print(f'FCFS {fcfs_avg[1]} | RR {rr_avg[1]} | SJF {sjf_avg[1]} | RandomAgent {ra_avg[1]}')
print()

print("Jain's Fairness Index ")
print(f'FCFS {fcfs_avg[2]} | RR {rr_avg[2]} | SJF {sjf_avg[2]} | RandomAgent {ra_avg[2]}')
print()

data = {
        "fcfs_avg": fcfs_avg,
        "rr_avg": rr_avg,
        "sjf_avg": sjf_avg,
        "ra_avg": ra_avg
    }
with open("saved_baseline_data.pkl", "wb") as f:
    pickle.dump(data, f)

