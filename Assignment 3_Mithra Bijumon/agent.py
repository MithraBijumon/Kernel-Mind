import numpy as np

class Agent:
    def __init__(self):
        self.alpha = 0.1
        self.training_episodes = 19000
        self.gamma = 0.9
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.queue_bins = 5
        self.queue_bin_size = 20
        self.wait_bins = 5
        self.wait_bin_size = 10
        self.burst_bins = 5
        self.burst_bin_size = 10
        self.num_actions = 3
        self.q_table = np.zeros((self.queue_bins, self.burst_bins, self.wait_bins, self.num_actions))

    def choose_action(self, state):
        prob = np.random.choice([0,1], p=[self.epsilon, 1-self.epsilon])
        if (prob==0):
            action = np.random.choice([0,1,2])
        else:
            action = np.argmax(self.q_table[state[0]][state[1]][state[2]])
        return action
    
    def learn(self, state, action, reward, next_state, done):
        qi_bin, bi_bin, wi_bin = state
        qf_bin, bf_bin, wf_bin = next_state
        if done:
            target = reward
        else:
            best_next_q = np.max(self.q_table[qf_bin][bf_bin][wf_bin])
            target = reward + self.gamma * best_next_q
        self.q_table[qi_bin][bi_bin][wi_bin][action] += self.alpha * (target - self.q_table[qi_bin][bi_bin][wi_bin][action])
