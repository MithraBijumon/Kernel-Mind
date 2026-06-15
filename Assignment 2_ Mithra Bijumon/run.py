import numpy as np
import os
import time
import matplotlib.pyplot as plt
from env import ProbeEnv
from agent import ProbeAgent

def render_probe_ascii(h, max_h, v, action, wind, step_count, is_jupyter=False):

    term_lines = 40

    if h > 150.0:
        display_max = max_h
        zoom_str = "[ CAMERA: WIDE ANGLE (1000m) ]"
    else:
        display_max = 150.0
        zoom_str = "[ CAMERA: TARGET APPROACH (150m) ]"

    pos = int((h / display_max) * term_lines)
    pos = max(0, min(term_lines, pos))

    wind_strs = ["~ Calm ~", "Gusty", "Adrian Gale"]
    thrust_str = "[####] ON" if action == 1 else "[    ] OFF"

    os.system('cls' if os.name == 'nt' else 'clear')

    print(f"T+{step_count:03d} | ALT: {h:6.1f}m | VEL: {v:7.1f}m/s | THRUST: {thrust_str} | WIND: {wind_strs[wind]}")
    print(zoom_str)
    print("-" * 75)

    for i in range(term_lines, -1, -1):
        if i == pos:
            if action == 1:
                print("        /\\")
                print("        ||")
                print("       /WW\\")
                print("        ||")
            else:
                print("        /\\")
                print("        ||")
                print("       /--\\")
                print("        ")
        else:
            if i % 10 == 0:
                print(f"{int((i/term_lines)*display_max):4d}m +----------------")
            else:
                print("      |")

    print("================== [ TAUMOEBA TARGET ] ==================")

    time.sleep(0.04)

def run_greedy_episode(env, agent):
    current_state = agent.discretize_state(env.reset())
    done = False
    total_reward = 0
    while not done:
        action = agent.choose_action(current_state)
        next_state, reward, done, info = env.step(action)
        next_state = agent.discretize_state(next_state)
        total_reward += reward
        current_state = next_state
        render_probe_ascii(
            h=next_state[0],
            max_h=1000,
            v=next_state[1],
            action=action,
            wind=next_state[2],
            step_count=env.steps
        )
    result = 'Timeout' if info['timeout'] else 'Crash' if info['crash'] else 'Success'
    return total_reward, result

if __name__ == "__main__":
    env = ProbeEnv()
    agent = ProbeAgent()
    agent.q_table = np.load('q_table.npy')
    episodes = 1
    agent.epsilon = 0.0
    total_reward, result = run_greedy_episode(env, agent)
    print(f"Total Reward: {total_reward}, Result: {result}")
