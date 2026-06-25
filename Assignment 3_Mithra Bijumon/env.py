import numpy as np
from process import Process
from workload_generator import workload_generator
from scheduling_policies import SJF, FCFS, RR

class Environment:
    def __init__(self, time_quantum=10):
        self.steps = 0
        self.max_steps = 2000
        self.workload = workload_generator(100)
        self.processes = self.workload.processes
        self.time = 0
        self.time_quantum = time_quantum
        self.rr_queue = [p for p in self.processes if p.arrival_time <= self.time]
        self.queue_bins = 5
        self.queue_bin_size = 20
        self.wait_bins = 5
        self.wait_bin_size = 10
        self.burst_bins = 5
        self.burst_bin_size = 10
        self.state = self.discretize_state(self.get_state())

    def avg_burst_time(self, processes):
        active = [p for p in processes if p.finish_time is None]
        if not active:
            return 0
        return sum(p.remaining_time for p in active) / len(active)
    
    def avg_wait_time(self, processes):
        active = [p for p in processes if p.finish_time is None]
        if not active:
            return 0
        return sum(p.waiting_time for p in active) / len(active)

    def get_queue_length(self, processes):
        return len([p for p in processes if p.arrival_time <= self.time and p.finish_time is None])
    
    def get_state(self):
        return [
            self.get_queue_length(self.processes),
            self.avg_burst_time(self.processes),
            self.avg_wait_time(self.processes)
        ]
    
    def discretize_state(self, state):
        queue_bucket = int(min(state[0]//self.queue_bin_size, self.queue_bins-1))
        burst_bucket = int(min(state[1]//self.burst_bin_size, self.burst_bins-1))
        wait_bucket = int(min(state[2]//self.wait_bin_size, self.wait_bins-1))
        return (queue_bucket, burst_bucket, wait_bucket)
    
    def reset(self):
        self.workload = workload_generator(100)
        self.processes = self.workload.processes
        self.time = 0
        self.steps = 0
        self.rr_queue = [p for p in self.processes if p.arrival_time <= self.time]
        self.state = self.discretize_state(self.get_state())
        return self.state
    
    def step(self, action):
        self.steps += 1
        reward = 0
        isRR = False
        if action == 0:
            agent = SJF(self.processes, self.time)
        elif action == 1:
            agent = FCFS(self.processes, self.time)
        elif action == 2:
            isRR = True
            agent = RR(self.processes, self.time, self.rr_queue)
        if isRR:
            self.rr_queue, process = agent.schedule()
        else:
            process = agent.schedule()
        if process:
            if process.start_time is None:
                process.start_time = self.time
            if process.remaining_time > self.time_quantum:
                self.time += self.time_quantum
                process.remaining_time -= self.time_quantum
                elapsed_time = self.time_quantum
            else:
                self.time += process.remaining_time
                elapsed_time = process.remaining_time
                process.remaining_time = 0
            if process.remaining_time == 0 and process.finish_time is None:
                process.finish_time = self.time
            for p in self.processes:
                if isRR and p.arrival_time<=self.time and p.finish_time is None and p not in self.rr_queue and p is not process:
                    self.rr_queue.append(p)
                if p is not process and p.arrival_time<=self.time and p.finish_time is None:
                    p.waiting_time += min(elapsed_time, self.time - p.arrival_time)
            if process.remaining_time > 0 and isRR:
                self.rr_queue.append(process)
        else:
            self.time += 1
            reward -= 1
        avg_wait = self.avg_wait_time(self.processes)
        starved = sum(p.remaining_time*(max(0, (p.waiting_time - 50)**2)) for p in self.processes)
        reward -= (avg_wait + 0.25 * starved)
        self.state = self.discretize_state(self.get_state())
        done = (
            all(p.finish_time is not None for p in self.processes)
            or self.steps >= self.max_steps
        )
        if self.steps >= self.max_steps:
            reward -= 100
        return self.state, reward, done

