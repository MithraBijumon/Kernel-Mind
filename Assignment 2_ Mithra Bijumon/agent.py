import numpy as np

class ProbeAgent:
    def __init__(self):
        self.h_bins = np.linspace(0, 1000, 50) 
        self.u_bins = np.linspace(-100, 100, 50) 
        self.q_table = np.zeros((len(self.h_bins), len(self.u_bins), 3, 2))
        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = 1.0
        self.epsilon_decay = 0.999

    def discretize_state(self, raw_state):
        hi, u, wind_idx = raw_state
        hi_bin = np.clip((np.digitize(hi, self.h_bins) - 1), 0, len(self.h_bins) - 1)
        u_bin = np.clip((np.digitize(u, self.u_bins) - 1), 0, len(self.u_bins) - 1)
        wind_bin = wind_idx
        return (hi_bin, u_bin, wind_bin)
    
    def choose_action(self, state):
        prob = np.random.choice([0, 1], p=[self.epsilon, 1 - self.epsilon])  
        if (prob == 0):
            action = np.random.choice([0, 1]) 
        else:
            hi_bin, u_bin, wind_bin = state
            action = np.argmax(self.q_table[hi_bin][u_bin][wind_bin])
        return action
    
    def learn(self, state, action, reward, next_state, done):
        hi_bin, u_bin, wind_bin = state
        hf_bin, v_bin, next_wind_bin = next_state
        if done:
            target = reward
        else:
            best_next_q = np.max(self.q_table[hf_bin][v_bin][next_wind_bin])
            target = reward + self.gamma * best_next_q
        self.q_table[hi_bin][u_bin][wind_bin][action] += self.alpha * (target - self.q_table[hi_bin][u_bin][wind_bin][action])
