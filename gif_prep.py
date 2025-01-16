import gymnasium as gym
from stable_baselines3 import PPO
from gymnasium import RewardWrapper
from PIL import Image  # Para criar o GIF
import numpy as np
import time  # Para medir o tempo de execução
from sb3_contrib import  TQC
import time


model = PPO.load("ppo_taxi")

# Run the episodes and capture frames
episodes = 1
all_frames = []  # To store the frames
max_steps = 200  # Maximum number of steps per episode
inicial = True

for ep in range(episodes):
    print(f"Starting episode {ep + 1}")
    obs, _ = env.reset()
    done = False
    episode_start_time = time.time()  # Record the start time of the episode
    episode_frames = []  # Collect frames for the current episode
    total_reward = 0
    steps = 0

   

    while not done and steps < max_steps:

      
        if(inicial):
            action, _ = model.predict(obs_inicial, deterministic=True)
            inicial = False
        else :
            action, _ = model.predict(obs, deterministic=True)

        action = int(action)  # Converter ação para inteiro, se necessário
        obs, reward, done, _, _ = env.step(action)
        total_reward += reward
        steps += 1


        # Capture the current frame
        frame = env.render()  # This will give an RGB array (height, width, 3)
        frame = Image.fromarray(frame)  # Convert numpy array to PIL Image
        episode_frames.append(frame)  # Append to the episode frame list

        time.sleep(0.5)

    # Add frames from this episode to the main list if it has frames
    if episode_frames:
        all_frames.extend(episode_frames)


# Save the captured frames as a GIF
if all_frames:  # Ensure we have frames captured
    output_path = "gifs/PPO_action.gif"
    all_frames[0].save(output_path, save_all=True, append_images=all_frames[1:], duration=25, loop=0)
    print(f"GIF saved to {output_path}")
else:
    print("No frames were captured. GIF not created.")

env.close()
