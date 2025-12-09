import gym
from gym.wrappers import GrayScaleObservation, ResizeObservation
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
import os

# Must match the wrapper class in train.py
class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = False
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info

def create_env():
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env, keep_dim=True)
    env = ResizeObservation(env, shape=84)
    return env

def main():
    # Load the environment
    env = DummyVecEnv([lambda: create_env()])
    env = VecFrameStack(env, n_stack=4, channels_order='last')

    # Load the trained model
    model_path = 'this_is_mario_model'
    if not os.path.exists(model_path + ".zip"):
        print(f"Model {model_path} not found. Please run train.py first.")
        return

    model = PPO.load(model_path)

    # Play
    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()

if __name__ == '__main__':
    main()


