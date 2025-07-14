
# from pathlib import Path
# import os
# CURRENT_DIR = Path(os.path.abspath('')).resolve()
# LOG_DIR = CURRENT_DIR.parent / f"trains/take-cover"
# LOG_DIR = str(LOG_DIR)
# print(LOG_DIR)
import os
import cv2
import numpy as np
from vizdoom import DoomGame, Mode, Button, GameVariable, ScreenResolution
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecTransposeImage
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common import callbacks
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import get_linear_fn
from gymnasium import RewardWrapper
from pathlib import Path

from typing import Callable
import vizdoom.gymnasium_wrapper
from utils import plot_rewards
from params import DQN_PARAMS, PPO_PARAMS
CURRENT_DIR = Path(os.path.abspath('')).resolve()
MODEL_PATH = str(CURRENT_DIR.parent / f"trains/health-gathering/ppo-Seba-v4-2/models/best_model")
agent = PPO.load(MODEL_PATH, print_system_info=True)
LOG_DIR = CURRENT_DIR.parent / f"trains/health-gathering/ppo-Seba-v4-2"
agent.save(str(LOG_DIR / f"saves/ppo_vizdoom"))
if not os.path.exists(str(LOG_DIR / f"policy")): os.makedirs(str(LOG_DIR / f"policy"))
agent.policy.save(str(LOG_DIR / f"policy/pesos.zip"))

