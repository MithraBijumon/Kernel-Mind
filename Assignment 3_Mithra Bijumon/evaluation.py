import numpy as np

def mean_wait_time(processes):
    total_wait_time = 0
    for p in processes:
        wait_time = p.waiting_time
        total_wait_time += wait_time
    return total_wait_time / len(processes)

def p90_wait_time(processes):
    wait_times = []
    for p in processes:
        wait_time = p.waiting_time
        wait_times.append(wait_time)
    return np.percentile(wait_times, 90)

def jain_fairness_index(processes):
    wait_times = []
    for p in processes:
        wait_time = p.waiting_time
        wait_times.append(wait_time)
    sum_wait_times = sum(wait_times)
    sum_wait_times_squared = sum(wt ** 2 for wt in wait_times)
    if (sum_wait_times_squared == 0 or len(wait_times) == 0):
        return 1.0
    j = (sum_wait_times ** 2) / (len(wait_times) * sum_wait_times_squared)
    return j