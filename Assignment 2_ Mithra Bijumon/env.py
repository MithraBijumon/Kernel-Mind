import numpy as np

class ProbeEnv:

    def __init__ (self):
        self.steps = 0
        self.max_steps = 1000
        self.m = 1000
        self.g = 13.7
        self.radrian = 10700000
        self.kdrag = 2
        self.fthrust = 25000
        self.dt = 0.1
        self.co2 = 0.91
        self.methane = 0.07
        self.argon = 0.01
        self.state = [1000, 0, 0]
        self.tpm = [[0.8, 0.2, 0.0], [0.2, 0.6, 0.2], [0.0, 0.3, 0.7]]

    def reset(self):
        self.state = [1000, 0, 0]
        self.steps = 0
        return tuple(self.state)
    
    def step(self, action):
        if (action == 1):
            thrust = self.fthrust
        else:
            thrust = 0
        hi = self.state[0]
        u = self.state[1]
        wind_idx_i = self.state[2]
        wind_idx_f = np.random.choice([0, 1, 2], p=self.tpm[wind_idx_i])
        if (wind_idx_i == 0):
            multiplier = 1
        elif (wind_idx_i == 1):
            multiplier = 1.5
        else:
            multiplier = 2.5
        drag = self.kdrag * u * u * np.sign(-u) * multiplier
        gravity = -self.m * self.g * (1 - hi / self.radrian)
        fnet = thrust + drag + gravity
        a = fnet / self.m
        v = u + a * self.dt
        hf = hi + v * self.dt
        self.state = [hf, v, wind_idx_f]
        reward = 0
        crash = False
        timeout = False
        self.steps += 1
        done = hf <= 0
        # Engine Burn
        if (action == 1):
            reward -= 1
        # Runaway Probe
        if self.steps >= self.max_steps:
            reward -= 100
            done = True
            timeout = True
        # Soft Target Catch
        if (hf<=0 and v >= -3):
            reward += 1000
        # Crushed by Pressure
        elif (hf<=0 and v < -3):
            crash = True
            reward -= 100*(abs(v)-3)
        return tuple(self.state), reward, done, {'crash': crash, 'timeout': timeout}