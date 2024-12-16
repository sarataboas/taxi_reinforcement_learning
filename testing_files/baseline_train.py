from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy 
import gymnasium as gym

# Create the environment
env = gym.make('Taxi-v3', render_mode= None)

# Reset the environment
obs, info = env.reset()

# Initialize the PPO model
model = PPO("MlpPolicy", env, verbose=1)

# Train the model (this will replace the loop with random actions)
model.learn(total_timesteps=1000000)  # Define the number of timesteps to train the model

# Save the trained model
model.save("ppo_taxi_model")


del model

model = PPO.load("ppo_taxi_model")

# After training, you can test the model by running it in the environment
obs, info = env.reset()


mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes = 100, deterministic = True)
print(f"Mean reward {mean_reward}; Standard Deviation reward {std_reward}")

# Run a loop to test the trained model
num_steps = 100  # Define the number of steps for testing
for step in range(num_steps):
    
    action, _states = model.predict(obs)  # Use the trained model to predict actions
    print(action)
    action = int(action)
    obs, reward, done, truncated, info = env.step(action)  # Step through the environment

    print(f"Step {step + 1}:")
    print(f"Action taken: {action}")
    print(f"New state: {obs}")
    print(f"Reward: {reward}")
    print(f"Done: {done}")
    print(f"Truncated: {truncated}\n")
    env.render()

    if done or truncated:
        print("Episode finished.")
        break

# Close the environment
env.close()
