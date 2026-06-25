import matplotlib.pyplot as plt
import numpy as np
import pickle

with open ('saved_rl_agent_data.pkl', 'rb') as f:
    rl_agent_data = pickle.load(f)

with open ('saved_baseline_data.pkl', 'rb') as f:
    baseline_data = pickle.load(f)