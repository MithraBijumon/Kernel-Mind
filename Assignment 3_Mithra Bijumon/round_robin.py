import numpy as np
from process import Process
from workload_generator import workload_generator

class round_robin:
    def __init__(self, processes, time_quantum):
        self.processes = processes
        self.time_quantum = time_quantum
        self.time = 0
        self.completed_processes = []
    def schedule(self):
        queue = []
        while len(self.completed_processes) < len(self.processes):
            for p in self.processes:
                if p.arrival_time <= self.time and p not in queue and p not in self.completed_processes:
                    queue.append(p)
            if queue:
                current_process = queue.pop(0)
                if current_process.start_time is None:
                    current_process.start_time = self.time
                if current_process.remaining_time > self.time_quantum:
                    self.time += self.time_quantum
                    current_process.remaining_time -= self.time_quantum
                    for p in self.processes:
                        if p.arrival_time <= self.time and p not in queue and p not in self.completed_processes and p != current_process:
                            queue.append(p)
                    queue.append(current_process)
                else:
                    self.time += current_process.remaining_time
                    current_process.finish_time = self.time
                    current_process.remaining_time = 0
                    self.completed_processes.append(current_process)
                    current_process.waiting_time = current_process.finish_time - current_process.arrival_time - current_process.burst_time
            else:
                self.time += 1

if __name__ == "__main__":
    num_processes = 10
    time_quantum = 10
    workload = workload_generator(num_processes)
    agent = round_robin(workload.processes, time_quantum)
    agent.schedule()
    for p in agent.completed_processes:
        print(f"Process with arrival time {p.arrival_time} and burst time {p.burst_time} finished at time {p.finish_time}")
                    