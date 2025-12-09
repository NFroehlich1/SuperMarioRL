import os
import gym
from gym.wrappers import GrayScaleObservation, ResizeObservation, FrameStack
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

# 1. Setup Environment Wrappers
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

class TrainAndLoggingCallback(BaseCallback):
    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls))
            self.model.save(model_path)
        return True

def create_env():
    # Define the environment
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    # Limit the action-space to simpler movements (Right, Right+A, etc)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    # Skip frames (process every 4th frame)
    env = SkipFrame(env, skip=4)
    # Convert to grayscale to reduce dimensionality
    env = GrayScaleObservation(env, keep_dim=True)
    # Resize to 84x84 (standard for Deep RL)
    env = ResizeObservation(env, shape=84)
    # Stack frames is handled by VecFrameStack usually, but can be done here.
    # We will do it via VecFrameStack in main for compatibility.
    return env

def main():
    # Create directories
    CHECKPOINT_DIR = './train/'
    LOG_DIR = './logs/'

    # Setup Environment
    # Wrap in DummyVecEnv for SB3
    env = DummyVecEnv([lambda: create_env()])
    # Stack 4 frames
    env = VecFrameStack(env, n_stack=4, channels_order='last')

    # Setup Model (PPO)
    model = PPO(
        'CnnPolicy', 
        env, 
        verbose=1, 
        tensorboard_log=LOG_DIR, 
        learning_rate=0.000001, 
        n_steps=512
    )

    # Setup Callback
    callback = TrainAndLoggingCallback(check_freq=10000, save_path=CHECKPOINT_DIR)

    # Train
    print("Starting training...")
    # Train for a manageable amount of steps for a demo (e.g. 100k or 1M)
    # Usually needs 1M+ for good results.
    model.learn(total_timesteps=100000, callback=callback)
    
    # Save final model
    model.save('this_is_mario_model')
    print("Training finished and model saved.")

if __name__ == '__main__':
    main()


