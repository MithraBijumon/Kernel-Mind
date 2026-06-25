import numpy as np
from process import Process
from workload_generator import workload_generator

class random_agent:
    def __init__(self, processes):
        self.processes = processes
        self.time = 0
        self.completed_processes = []
    def schedule(self):
        while len(self.completed_processes) < len(self.processes):
            available_processes = [p for p in self.processes if p.arrival_time <= self.time and p not in self.completed_processes]
            if available_processes:
                process = np.random.choice(available_processes)
                if process.start_time is None:
                    process.start_time = self.time
                process.remaining_time -= 1
                if process.remaining_time == 0:
                    process.finish_time = self.time + 1
                    self.completed_processes.append(process)
                    process.waiting_time = process.finish_time - process.arrival_time - process.burst_time
            self.time += 1

if __name__ == "__main__":
    num_processes = 10
    workload = workload_generator(num_processes)
    agent = random_agent(workload.processes)
    agent.schedule()
    for process in agent.completed_processes:
        print(f"Process with arrival time {process.arrival_time} and burst time {process.burst_time} finished at time {process.finish_time}")