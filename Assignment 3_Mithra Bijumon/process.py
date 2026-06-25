import numpy as np

class Process:
    def __init__(self, data):
        self.data = data
        self.arrival_time = data["arrival_time"]
        self.burst_time = data["burst_time"]
        self.remaining_time = data["burst_time"]
        self.start_time = None
        self.finish_time = None
        self.waiting_time = 0