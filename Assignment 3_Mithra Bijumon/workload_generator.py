import numpy as np
from process import Process

class workload_generator:
    def __init__(self, num_processes):
        self.num_processes = num_processes
        self.processes = self.generate_processes()
    def generate_processes(self):
        processes = []
        for i in range(self.num_processes):
            arrival_time = np.random.randint(0,360)
            if np.random.rand() < 0.8:
                burst_time = np.random.randint(1,15)
            else:
                burst_time = np.random.randint(50,150)
            data = {"arrival_time": arrival_time, "burst_time": burst_time}
            processes.append(Process(data))
        return processes