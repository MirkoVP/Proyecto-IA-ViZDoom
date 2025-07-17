import os
import cv2
import numpy as np
import gymnasium as gym
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecTransposeImage, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common import callbacks
from gymnasium import RewardWrapper
from pathlib import Path

import vizdoom.gymnasium_wrapper

ENV = "VizdoomHealthGathering-v0"
#ENV = "VizdoomTakeCover-v0"
RESOLUTION = (60, 45)

model = "dqn"
num = "Seba-v4-3"
map = "health-gathering"
#map = "take-cover"
CURRENT_DIR = Path(os.path.abspath('')).resolve()
#MODEL_PATH = f"trains/{map}/{model}-{num}/saves/{model}_vizdoom"
MODEL_PATH = str(CURRENT_DIR.parent / f"trains/{map}/{model}-{num}/models/best_model")

class RewardShapingWrapper(RewardWrapper):
    def __init__(self, env, above_70_reward=1, healings_rewards=10):
        super(RewardShapingWrapper, self).__init__(env)
        self.above_70_reward = above_70_reward
        self.healings_rewards = healings_rewards
        self.previous_heal_count = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        game_variables = self.env.unwrapped.game.get_state().game_variables

        self.previous_heal_count = game_variables[1]  # ITEMCOUNT

        #print(f"Game variables: {game_variables}")
        #print(f"Reset: Damage taken: {self.previous_damage_taken}, Hitcount: {self.previous_hitcount}, Ammo: {self.previous_ammo}")

        return obs, info

    def reward(self, reward):
        #print(f"Reward original: {reward}")
        custom_reward = reward
        game_state = self.env.unwrapped.game.get_state()

        if game_state:
            game_variables = game_state.game_variables
            current_health = game_variables[0]  # HEALTH
            current_heal_count = game_variables[1] # ITEMCOUNT

            # recompensa por tener sobre 70 de vida
            if current_health >= 70:
                custom_reward += self.above_70_reward
            #Recompensa por recoger heals
            if current_heal_count > self.previous_heal_count:
                heal_count_delta = current_heal_count - self.previous_heal_count
                reward_gain = heal_count_delta * self.healings_rewards
                custom_reward += reward_gain
            self.previous_heal_count = current_heal_count

        return custom_reward

class ObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, shape=RESOLUTION):
        super().__init__(env)
        self.image_shape = shape
        self.image_shape_reverse = shape[::-1]

        # Create new observation space with the new shape
        num_channels = env.observation_space["screen"].shape[-1]
        new_shape = (shape[0], shape[1], num_channels)
        self.observation_space = gym.spaces.Box(
            0, 255, shape=new_shape, dtype=np.uint8
        )

    def observation(self, observation):
        observation = cv2.resize(observation["screen"], self.image_shape_reverse)
        return observation


if __name__ == "__main__":

    def wrap_env(env):
        env = ObservationWrapper(env)
        env = RewardShapingWrapper(env)
        #env = gym.wrappers.TransformReward(env, lambda r: r * 0.001)
        return env

    env = make_vec_env(ENV, wrapper_class=wrap_env, env_kwargs={"frame_skip": 4, "render_mode": "human"})

    # Load the trained agent
    if model == "dqn":
        agent = DQN.load(MODEL_PATH, print_system_info=True)
    else:
        agent = PPO.load(MODEL_PATH, print_system_info=True)

    os.system("clear")

    # print env and num
    print(f"Env: {ENV}")
    print(f"Model: {model}")
    print(f"Num: {num}")

    episode = 0
    for _ in range(10):
        obs = env.reset()
        done = False
        total_reward = 0
        episode+=1
        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            #print(f"Reward: {reward}")
            total_reward += reward
            env.render()

        print(f"Episode: {episode} Reward: {total_reward}")

    env.close()


