from utils import plot_comparison, plot_side_by_side, plot_rewards, plot_epsilon, plot_results
from pathlib import Path
import os

# Plot comparison between two models
path = "trains/deathmatch"
CURRENT_DIR = Path(os.path.abspath('')).resolve()
LOG_PATH = CURRENT_DIR / "trains/take-cover/dqn-Seba-3"
#plot_comparison(f"{path}dqn-5", f"{path}ppo-6", window=10)
#plot_side_by_side(f"{path}dqn-1", f"{path}ppo-1", window=10)
plot_rewards(LOG_PATH, False, window=10)

#plot_epsilon(f"{path}dqn-6")

# Plot results for all models
#plot_results(path)