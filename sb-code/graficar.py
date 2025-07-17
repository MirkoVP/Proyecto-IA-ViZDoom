from utils import plot_comparison, plot_side_by_side, plot_rewards, plot_epsilon, plot_results_mod
from pathlib import Path
import os

# Plot comparison between two models
path = "trains/deathmatch"
CURRENT_DIR = Path(os.path.abspath('')).resolve()
LOG_PATH = CURRENT_DIR.parent / "trains/health-gathering"
#LOG_PATH = CURRENT_DIR.parent / "trains/take-cover"
#plot_comparison(str(LOG_PATH / "dqn-Seba-v4-3"), str(LOG_PATH / "ppo-Seba-v4-3"), window=10)
#plot_side_by_side(str(LOG_PATH / "dqn-Seba-v4-4"), str(LOG_PATH / "ppo-Seba-v4-4"), window=10)
#plot_rewards(str(LOG_PATH), False, window=10)

#plot_epsilon(f"{path}dqn-6")

# Plot results for all models
plot_results_mod(LOG_PATH)