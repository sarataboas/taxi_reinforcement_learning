##### Avaliação com evaluate_policy

from stable_baselines3.common.evaluation import evaluate_policy

# Carregar o modelo treinado
model = PPO.load("ppo_taxi_model")

# Avaliar o modelo em 100 episódios
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100, deterministic=True)

print(f"Recompensa média: {mean_reward}, Desvio padrão: {std_reward}")


###### Monitorização do Treino - avalia a evolução das recompensas ao longo do treino
from stable_baselines3.common.callbacks import EvalCallback

# Callback para avaliação durante o treinamento 
eval_callback = EvalCallback(
    env,
    best_model_save_path="./logs/",
    log_path="./logs/",
    eval_freq=10000,
    n_eval_episodes=10,
    deterministic=True,
    render=False,
)

# Treinar com o callback
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1000000, callback=eval_callback)


##### Learning Curve
import matplotlib.pyplot as plt

# Armazenar recompensas
rewards = []

obs = env.reset()
for _ in range(10000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    rewards.append(reward)
    if done:
        obs = env.reset()

# Plotar recompensas acumuladas
plt.plot(rewards)
plt.title("Recompensas Acumuladas")
plt.xlabel("Passos")
plt.ylabel("Recompensa")
plt.show()


#### Metricas complementares 
# 1. Taxa de sucesso
success_count = 0
n_episodes = 100

for _ in range(n_episodes):
    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
    if "is_success" in info and info["is_success"]:
        success_count += 1

success_rate = success_count / n_episodes
print(f"Taxa de sucesso: {success_rate * 100:.2f}%")

