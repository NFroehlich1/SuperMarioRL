import streamlit as st
import os
import gym
import numpy as np
gym.logger.set_level(40)  # Suppress warnings
from gym.wrappers import GrayScaleObservation, ResizeObservation
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import BaseCallback
import torch
import time

# --- Setup Environment & Wrappers ---
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

class RewardShaping(gym.Wrapper):
    """
    Advanced Reward Shaping:
    Belohnt NUR echten Fortschritt (neue Max-X-Position).
    Verhindert Reward-Hacking durch Hin-und-Herlaufen.
    """
    def __init__(self, env):
        super().__init__(env)
        self._x_position_max = 0
        self._steps_stuck = 0
        
    def reset(self, **kwargs):
        self._x_position_max = 0
        self._steps_stuck = 0
        return self.env.reset(**kwargs)
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        
        # Aktuelle Position
        x_pos = info.get('x_pos', 0)
        flag_get = info.get('flag_get', False)
        
        # Custom Reward initialisieren
        custom_reward = 0.0
        
        # 1. Progress Reward (NUR für neuen Boden)
        # Wenn wir weiter sind als je zuvor in dieser Episode:
        if x_pos > self._x_position_max:
            diff = x_pos - self._x_position_max
            custom_reward += diff * 1.0  # 1 Punkt pro neuem Pixel
            self._x_position_max = x_pos
            self._steps_stuck = 0 # Reset Stuck-Counter
        else:
            self._steps_stuck += 1
            
        # 2. Time Penalty (Konstanter Druck)
        custom_reward -= 0.05
        
        # 3. Stuck Penalty (Wenn er gegen Wände läuft)
        if self._steps_stuck > 100: # Wenn 100 Frames (ca 1.5s) kein Fortschritt
            custom_reward -= 0.5 # Zusätzliche Strafe pro Frame
            
        # 4. Death Penalty
        if done and not flag_get:
            custom_reward -= 50.0
            # Optional: Strafe reduzieren, wenn er weit gekommen ist? 
            # Nein, Tod ist immer schlecht.
            
        # 5. Level Completion
        if flag_get:
            custom_reward += 1000.0
            
        # Debug / Info ins Info-Dict schreiben (für Analyse)
        info['custom_reward'] = custom_reward
        info['max_x'] = self._x_position_max
        
        return obs, custom_reward, done, info

def create_env():
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = RewardShaping(env)  # Reward Shaping für besseres Lernen
    env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env, keep_dim=True)
    env = ResizeObservation(env, shape=84)
    return env

def get_vectorized_env(n_envs=1, use_subproc=False):
    """
    Erstellt vectorized environment.
    
    Args:
        n_envs: Anzahl paralleler Environments (mehr = schneller, aber mehr RAM)
        use_subproc: True für SubprocVecEnv (schneller, aber mehr Overhead)
    """
    if n_envs == 1:
        env = DummyVecEnv([lambda: create_env()])
    else:
        if use_subproc:
            # SubprocVecEnv: Jedes Env in separatem Prozess (besser für CPU-intensive Tasks)
            env = SubprocVecEnv([lambda: create_env() for _ in range(n_envs)])
        else:
            # DummyVecEnv: Alle Envs im selben Prozess (weniger Overhead, aber langsamer)
            env = DummyVecEnv([lambda: create_env() for _ in range(n_envs)])
    
    env = VecFrameStack(env, n_stack=4, channels_order='last')
    return env

def get_action_name(action_idx):
    """Gibt die gedrückten Tasten als String zurück."""
    if action_idx == 0: return "NOOP (Nichts)"
    if action_idx == 1: return "Rechts"
    if action_idx == 2: return "Rechts + A (Springen)"
    if action_idx == 3: return "Rechts + B (Rennen)"
    if action_idx == 4: return "Rechts + A + B (Rennen + Springen)"
    if action_idx == 5: return "A (Springen)"
    if action_idx == 6: return "Links"
    return f"Action {action_idx}"

# --- Callback for Progress ---
class StreamlitCallback(BaseCallback):
    def __init__(self, progress_bar, status_text, total_timesteps, save_path, verbose=1):
        super(StreamlitCallback, self).__init__(verbose)
        self.progress_bar = progress_bar
        self.status_text = status_text
        self.total_timesteps = total_timesteps
        self.save_path = save_path
        self.best_mean_reward = -float('inf')
        self.check_freq = 10000  # Check alle 10k Schritte
        # Eigene Episode-Reward-Verwaltung
        self.episode_rewards = []
        self.current_episode_reward = 0.0

    def _on_step(self):
        # Rewards und Dones aus den lokalen Variablen des Trainers holen
        rewards = self.locals.get("rewards")
        dones = self.locals.get("dones")

        # VecEnv: rewards/dones sind Arrays; wir nehmen Env 0
        if rewards is not None:
            try:
                r = float(rewards[0])
            except Exception:
                r = float(rewards)
            self.current_episode_reward += r

        if dones is not None:
            try:
                done_flag = bool(dones[0])
            except Exception:
                done_flag = bool(dones)
            if done_flag:
                self.episode_rewards.append(self.current_episode_reward)
                self.current_episode_reward = 0.0

        progress = min(self.num_timesteps / self.total_timesteps, 1.0)
        self.progress_bar.progress(progress)
        
        # Zeige Fortschritt und aktuellen Reward
        if len(self.episode_rewards) > 0:
            mean_reward = sum(self.episode_rewards[-100:]) / min(len(self.episode_rewards), 100)
            self.status_text.text(
                f"Training: {int(progress * 100)}% ({self.num_timesteps}/{self.total_timesteps}) | "
                f"Avg Reward: {mean_reward:.1f}"
            )
            
            # Speichere bestes Modell
            if self.num_timesteps % self.check_freq == 0:
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    if self.save_path:
                        self.model.save(f"{self.save_path}_best")
        else:
            self.status_text.text(f"Training Progress: {int(progress * 100)}% ({self.num_timesteps}/{self.total_timesteps} steps)")
        
        return True

# --- UI Layout ---
st.set_page_config(page_title="Super Mario RL Trainer", layout="wide")
st.title("🍄 Super Mario Bros RL Trainer")

st.markdown("""
Trainiere deinen eigenen Super Mario Agenten direkt im Browser!
Wähle einen Algorithmus, setze die Trainingsschritte und schau dem Agenten beim Spielen zu.
""")

# Sidebar for Settings
st.sidebar.header("⚙️ Einstellungen")
algo_choice = st.sidebar.selectbox("Algorithmus", ["PPO", "DQN", "A2C"])
total_timesteps = st.sidebar.number_input("Trainingsschritte", min_value=1000, max_value=10000000, value=500000, step=50000)
st.sidebar.caption("💡 Empfehlung: 500k+ für Level-Completion")

# Hardware-Optimierung
st.sidebar.header("🚀 Hardware-Optimierung")
n_envs = st.sidebar.slider("Parallele Environments", min_value=1, max_value=8, value=4, step=1)
st.sidebar.caption(f"💻 {n_envs} Envs nutzen {n_envs} CPU-Kerne parallel")
use_subproc = st.sidebar.checkbox("Subprocess-Modus (schneller, mehr RAM)", value=True)
st.sidebar.caption("⚠️ Subprocess: Schneller, aber mehr RAM-Verbrauch")

model_name = st.sidebar.text_input("Modell Name", value="mario_model")

# Paths
CHECKPOINT_DIR = './train_web/'
LOG_DIR = './logs_web/'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
model_path = os.path.join(CHECKPOINT_DIR, f"{model_name}_{algo_choice}")

# --- Tabs ---
tab1, tab2 = st.tabs(["🏋️ Training", "🎮 Play / Watch"])

# --- Tab 1: Training ---
with tab1:
    st.header(f"Training mit {algo_choice}")
    
    st.info("""
    🎯 **Level-Completion Strategie:**
    - **Mindestens 500.000 Schritte** für Level-Completion empfohlen
    - **PPO** ist meist am besten für Super Mario
    - Das beste Modell wird automatisch gespeichert (höchster Reward)
    - Training kann mehrere Stunden dauern - lass es laufen!
    """)
    
    # Option: Bestehendes Modell weiter trainieren
    continue_training = st.checkbox("Bestehendes Modell weiter trainieren?")
    load_model_path = None
    if continue_training:
        # Finde alle .zip Dateien im Checkpoint Dir
        model_files = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".zip")]
        if model_files:
            selected_model = st.selectbox("Wähle Modell zum Fortsetzen:", model_files)
            load_model_path = os.path.join(CHECKPOINT_DIR, selected_model)
            st.success(f"Lade Modell: {selected_model}")
        else:
            st.warning("Keine gespeicherten Modelle gefunden.")
            continue_training = False

    if st.button("Start Training"):
        # Prüfe verfügbare Hardware
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        st.info(f"🔧 Hardware: {device.upper()} | {n_envs} parallele Environments")
        
        env = get_vectorized_env(n_envs=n_envs, use_subproc=use_subproc)
        
        model = None
        
        if continue_training and load_model_path:
            st.write(f"🔄 Lade Modell von {load_model_path}...")
            custom_objects = {'learning_rate': 2.5e-4, 'clip_range': 0.2, 'n_steps': 2048} # Neue Hyperparameter erzwingen wenn nötig
            
            if algo_choice == "PPO":
                model = PPO.load(load_model_path, env=env, device=device, custom_objects=custom_objects)
            elif algo_choice == "DQN":
                model = DQN.load(load_model_path, env=env, device=device)
            elif algo_choice == "A2C":
                model = A2C.load(load_model_path, env=env, device=device)
            
            st.write("✅ Modell geladen! Training wird fortgesetzt.")
            
        else:
            st.write("🆕 Starte neues Training...")
            # Initialize Model mit optimierten Hyperparametern für Level-Completion
            if algo_choice == "PPO":
                model = PPO(
                    'CnnPolicy', 
                    env, 
                    verbose=1, 
                    tensorboard_log=LOG_DIR, 
                    learning_rate=2.5e-4,
                    n_steps=2048,  # Größere Batches für stabileres Lernen
                    batch_size=64,
                    n_epochs=10,  # Mehr Epochs für besseres Lernen
                    gamma=0.99,
                    gae_lambda=0.95,  # GAE Lambda für bessere Value-Schätzung
                    clip_range=0.2,
                    ent_coef=0.01,  # Exploration
                    vf_coef=0.5
                )
            elif algo_choice == "DQN":
                model = DQN(
                    'CnnPolicy', 
                    env, 
                    verbose=1, 
                    tensorboard_log=LOG_DIR, 
                    buffer_size=100000,  # Sehr großer Buffer
                    learning_starts=5000,  # Mehr Exploration vor Lernen
                    learning_rate=1e-4,
                    gamma=0.99,
                    exploration_fraction=0.1,  # 10% der Zeit für Exploration
                    exploration_final_eps=0.05,
                    target_update_interval=1000
                )
            elif algo_choice == "A2C":
                model = A2C(
                    'CnnPolicy', 
                    env, 
                    verbose=1, 
                    tensorboard_log=LOG_DIR,
                    learning_rate=7e-4,
                    gamma=0.99,
                    n_steps=5,  # Mehr Schritte für bessere Schätzung
                    ent_coef=0.01
                )
            
        # Progress Bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        callback = StreamlitCallback(progress_bar, status_text, total_timesteps, model_path)
        
        with st.spinner(f"Training {algo_choice} für {total_timesteps} Schritte..."):
            model.learn(total_timesteps=total_timesteps, callback=callback)
            model.save(model_path)
            
        st.success(f"Training abgeschlossen! Modell gespeichert unter: {model_path}")
        st.info("Du kannst nun zum 'Play / Watch' Tab wechseln, um das Ergebnis zu sehen.")

# --- Tab 2: Play ---
with tab2:
    st.header("Agent beim Spielen zusehen")
    
    # Check if model exists
    use_best = st.checkbox("Beste Modell-Version verwenden (höchster Reward)", value=True)
    model_to_load = f"{model_path}_best" if use_best else model_path
    
    if not os.path.exists(model_to_load + ".zip") and not os.path.exists(model_path + ".zip"):
        st.warning(f"Kein Modell gefunden. Bitte trainiere zuerst.")
    else:
        # Fallback auf normales Modell wenn bestes nicht existiert
        if not os.path.exists(model_to_load + ".zip"):
            model_to_load = model_path
            st.info("Beste Version nicht gefunden, verwende Standard-Modell.")
        
        if st.button("Start Game"):
            # Load Environment for Render
            # Note: We use a single environment for rendering, not vectorized for simplicity in loop
            # But model expects vectorized input shape usually, or we wrap it.
            # SB3 models predict on single observations if structured correctly, but usually expect vec_env wrapper style.
            
            env = get_vectorized_env()
            
            # Load Model
            if algo_choice == "PPO":
                model = PPO.load(model_to_load)
            elif algo_choice == "DQN":
                model = DQN.load(model_to_load)
            elif algo_choice == "A2C":
                model = A2C.load(model_to_load)
            
            obs = env.reset()
            
            # Streamlit Image Placeholder
            image_placeholder = st.empty()
            action_text = st.empty()  # Placeholder für Action-Text
            
            st.write("Rendering gameplay... (Interagiere mit der Seite, um zu stoppen)")
            
            # Run for a max number of steps to avoid infinite loops
            for _ in range(2000):
                action, _ = model.predict(obs)
                
                # Action Text anzeigen
                # action ist ein Array bei vector env, wir nehmen das erste Element
                act_idx = action[0] if isinstance(action, (list, tuple, np.ndarray)) else action
                action_text.markdown(f"### 🎮 Controller: **{get_action_name(act_idx)}**")
                
                obs, reward, done, info = env.step(action)
                
                # Render to array
                frame = env.render(mode='rgb_array')
                
                # Update Image in Streamlit
                image_placeholder.image(frame, channels="RGB", width=600)
                
                # Slow down slightly
                time.sleep(0.01)
                
                # done is an array in VecEnv
                if done[0]:
                    break
            
            env.close()

