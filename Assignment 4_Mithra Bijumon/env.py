import numpy as np
from process import Process
from workload_generator import workload_generator

class Environment:
    def __init__(self, time_quantum=10):
        self.steps = 0
        self.max_steps = 2000
        self.workload = workload_generator(num_processes=100)
        self.time_quantum = time_quantum
        self.processes = self.workload.processes
        self.time = 0
        self.rr_queue = [p for p in self.processes if p.arrival_time <= self.time]
        self.