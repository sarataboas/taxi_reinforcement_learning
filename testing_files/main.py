from gymnasium.utils.env_checker import check_env
from TaxiEnv import CustomTaxiEnv 

env = CustomTaxiEnv(grid_size=(10, 10), obstacles=[(1, 1), (2, 2)])
check_env(env, warn=True)


env = CustomTaxiEnv(grid_size=(10, 10), obstacles=[(1, 1), (2, 2)])
obs, info = env.reset()
print(f"Initial Observation: {obs}")

for step in range(20):
    action = env.action_space.sample()  # Random action
    obs, reward, done, info = env.step(action)
    print(f"Step: {step}, Action: {action}, Observation: {obs}, Reward: {reward}, Done: {done}")
    env.render()

    if done:
        print("Episode finished!")
        break

env.close()
