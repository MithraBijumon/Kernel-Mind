# README

## Requirements

Install the required Python packages:

```bash
pip install numpy matplotlib
```

## Project Files

* `env.py` – Adrian physics environment
* `agent.py` – Tabular Q-Learning agent
* `train.py` – Trains the agent and plots learning curves
* `run.py` - Runs a single greedy evaluation episode using the trained Q-table

## Running the Project

To train the agent:

```bash
python train.py
```

The script will:

* Train the Q-Learning agent for 20,000 episodes.
* Display and save the moving average reward graph.
* Display and save the moving average landing success rate graph.
* Print the total number of Successes, Crashes, and Timeouts.
* Save the learned Q-table

To evaluate the trained agent:

```bash
python run.py
```
The script will:

* Load the saved q_table.npy, set ε = 0.0, and run one purely greedy episode while printing the probe's altitude, velocity, and wind state at each step.

* At the end of the run, the total reward and final outcome (Success, Crash, or Timeout) are displayed.

## Output

After training, you may save:

* The trained Q-table:

  ```python
  np.save("q_table.npy", agent.q_table)
  ```
* The training plots:

  ```python
  plt.savefig("learning_curve.png")
  ```

## Evaluation

To evaluate the trained policy, load the saved Q-table, set:

```python
agent.epsilon = 0.0
```

and run a single greedy episode while calling `render_probe_ascii()` at each step to observe the probe's descent.
