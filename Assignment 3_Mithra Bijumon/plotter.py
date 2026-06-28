import matplotlib.pyplot as plt
import numpy as np
import pickle

with open ('saved_rl_agent_data.pkl', 'rb') as f:
    rl_agent_data = pickle.load(f)

with open ('saved_baseline_data.pkl', 'rb') as f:
    baseline_data = pickle.load(f)

moving_mean_wait_times = rl_agent_data["moving_mean_wait_times"]
moving_p90_wait_times = rl_agent_data["moving_p90_wait_time"]
moving_jfis = rl_agent_data["moving_jfs"]
episodes = np.arange(len(moving_mean_wait_times))

fcfs_avg = baseline_data["fcfs_avg"]
rr_avg = baseline_data["rr_avg"]
ra_avg = baseline_data["ra_avg"]
sjf_avg = baseline_data["sjf_avg"]

plt.figure(figsize=(10,6))
plt.plot(episodes, moving_mean_wait_times, label="RL Agent", linewidth=2)
plt.axhline(y=fcfs_avg[0], linestyle='--', label=f'FCFS')
plt.axhline(y=rr_avg[0], linestyle='--', label=f'RR')
plt.axhline(y=sjf_avg[0], linestyle='--', label=f'SJF')
plt.axhline(y=ra_avg[0], linestyle='--', label=f'Random')
plt.xlabel("Training Episode")
plt.ylabel("Mean Wait Time")
plt.title("RL Agent Convergence vs Baselines")
plt.legend()
plt.grid(True)
plt.savefig('mean_wait_times_comparision.png')
plt.show()

plt.figure(figsize=(10,6))
plt.plot(episodes, moving_p90_wait_times, label="RL Agent")
plt.axhline(fcfs_avg[1], linestyle='--', label='FCFS')
plt.axhline(rr_avg[1], linestyle='--', label='RR')
plt.axhline(sjf_avg[1], linestyle='--', label='SJF')
plt.xlabel("Training Episode")
plt.ylabel("P90 Wait Time")
plt.title("P90 Wait Time Convergence")
plt.legend()
plt.savefig('p90_wait_times_comparision.png')
plt.show()

plt.figure(figsize=(10,6))
plt.plot(episodes, moving_jfis, label="RL Agent")
plt.axhline(fcfs_avg[2], linestyle='--', label='FCFS')
plt.axhline(rr_avg[2], linestyle='--', label='RR')
plt.axhline(sjf_avg[2], linestyle='--', label='SJF')
plt.xlabel("Training Episode")
plt.ylabel("Jain Fairness Index")
plt.title("Fairness Comparison")
plt.legend()
plt.savefig('Jain Fairness Index Comparision.png')
plt.show()