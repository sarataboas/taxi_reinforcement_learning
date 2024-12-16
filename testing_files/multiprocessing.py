from stable_baselines3 import PPO # PPO 2 - latest version of the algorithm
from stable_baselines3.common.evaluation import evaluate_policy 
import gymnasium as gym
from ignorav2 import CustomTaxiEnv
from gymnasium.wrappers import TimeLimit

# hyperameters tuning
from sklearn.model_selection import ParameterGrid
from concurrent.futures import ThreadPoolExecutor, as_completed



def env_fn():
    env = gym.make("Taxi-v3")
    env = TimeLimit(env, max_episode_steps=200)
    return env

def evaluate_params(model_class, param_comb, total_timesteps, n_eval_episodes, verbose):
    print(f"Testing with parameters: {param_comb}")
    env = env_fn()  # Cria uma nova instância do ambiente para cada thread
    model = model_class("MlpPolicy", env, **param_comb, verbose=verbose)
    model.learn(total_timesteps=total_timesteps)
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=n_eval_episodes, deterministic=True)
    print(f"Mean reward: {mean_reward} with parameters: {param_comb}")
    return mean_reward, param_comb


def param_tuning_multithread(model_class, params, total_timesteps, n_eval_episodes, verbose, max_threads):

    param_grid = list(ParameterGrid(params))  # Cria todas as combinações
    best_reward = -float('inf')
    best_params = None

    # Executor de threads
    with ThreadPoolExecutor(max_threads) as executor:
        futures = [
            executor.submit(evaluate_params, model_class, env_fn, param_comb, total_timesteps, n_eval_episodes, verbose)
            for param_comb in param_grid
        ]

        for future in as_completed(futures):
            try:
                mean_reward, param_comb = future.result()
                if mean_reward > best_reward:
                    best_reward = mean_reward
                    best_params = param_comb
            except Exception as e:
                print(f"Erro ao processar parâmetros: {e}")

    print(f"Best parameters: {best_params}")
    print (f"Best reward: {best_reward}")
    return best_params, best_reward




params_ppo_grid = {

    'learning_rate': [1e-4, 1e-3, 1e-2],
    'n_steps': [128, 256, 512],
    'batch_size': [32, 64],
    'ent_coef': [0.0, 0.01, 0.1],
    'gamma': [0.99, 0.95]

}


best_params, best_reward = param_tuning_multithread(PPO, params_ppo_grid, total_timesteps=100000, n_eval_episodes=10, verbose=0, max_threads=4)