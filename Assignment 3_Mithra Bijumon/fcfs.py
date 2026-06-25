import numpy as np
from process import Process
from workload_generator import workload_generator

class FCFS:
    def __init__(self, processes, time_quantum=None):
        self.processes = processes
        self.time_quantum = time_quantum
        self.time = 0
        self.completed_processes = []

    def schedule(self):
        while len(self.completed_processes) < len(self.processes):
            available_processes = [p for p in self.processes if p.arrival_time <= self.time and p not in self.completed_processes]
            if available_processes:
                process = min(available_processes, key=lambda p: p.arrival_time)
                if process.start_time is None:
                    process.start_time = self.time
                process.remaining_time = 0
                self.time += process.burst_time
                if process.remaining_time == 0:
                    process.finish_time = self.time
                    self.completed_processes.append(process)
                    process.waiting_time = process.finish_time - process.arrival_time - process.burst_time
            else:
                start_time = self.time
                self.time = min([p.arrival_time for p in self.processes if p not in self.completed_processes])
    
if __name__ == "__main__":
    num_processes = 10
    workload = workload_generator(num_processes)
    agent = FCFS(workload.processes)
    agent.schedule()
    for process in agent.completed_processes:
        print(f"Process with arrival time {process.arrival_time} and burst time {process.burst_time} finished at time {process.finish_time}")
