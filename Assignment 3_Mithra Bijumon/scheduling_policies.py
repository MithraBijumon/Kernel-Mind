import numpy as np
from process import Process

class FCFS:
    def __init__(self, processes, time):
        self.processes = processes
        self.time = time
    def schedule(self):
        available_processes = [p for p in self.processes if p.arrival_time <= self.time and p.remaining_time > 0]
        if available_processes:
            process = min(available_processes, key=lambda p: p.arrival_time)
        else:
            process = None
        return process

class SJF:
    def __init__(self, processes, time):
        self.processes = processes
        self.time = time
    def schedule(self):
        available_processes = [p for p in self.processes if p.arrival_time <= self.time and p.remaining_time > 0]
        if available_processes:
            process = min(available_processes, key=lambda p: p.remaining_time)
        else:
            process = None
        return process

class RR:
    def __init__(self, processes, time, rr_queue):
        self.processes = processes
        self.time = time
        self.rr_queue = rr_queue
    def schedule(self):
        if self.rr_queue:
            process = self.rr_queue.pop(0)
        else:
            process = None
        return (self.rr_queue, process)