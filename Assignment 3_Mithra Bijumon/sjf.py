import numpy as np
from process import Process
from workload_generator import workload_generator

class SJF:
    def __init__(self, processes):
        self.processes = processes
        self.time = 0
        self.completed_processes = []
    def schedule(self):
        while len(self.completed_processes) < len(self.processes):
            curr_process = None
            available_processes = [p for p in self.processes if p.arrival_time <= self.time and p not in self.completed_processes]
            if available_processes:
                curr_process = min(available_processes, key = lambda p: p.burst_time)
                if curr_process.start_time is None:
                    curr_process.start_time = self.time
                curr_process.remaining_time = 0
                self.time += curr_process.burst_time
                curr_process.finish_time = self.time
                self.completed_processes.append(curr_process)
                curr_process.waiting_time = curr_process.finish_time - curr_process.arrival_time - curr_process.burst_time
            else:
                self.time += 1

if __name__ == "__main__":
    num_processes = 10
    workload = workload_generator(num_processes)
    agent = SJF(workload.processes)
    agent.schedule()
    for process in agent.completed_processes:
        print(f"Process with arrival time {process.arrival_time} and burst time {process.burst_time} finished at time {process.finish_time}")



            