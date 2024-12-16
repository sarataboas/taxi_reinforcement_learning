from gymnasium import RewardWrapper, ActionWrapper
import gymnasium

class CustomRewardWrapper(RewardWrapper):
    def __init__(self, env, penalty=1.0):
        super().__init__(env)
        self.env = env
        self.visited_positions = set()  # Set to store visited positions
        self.penalty = penalty

    def step(self, action):
        obs, reward, done, _, info = self.env.step(action)

        # Get the taxi's position (assuming obs contains the position as (x, y) coordinates)
        taxi_pos = (obs[0], obs[1])  # The position is stored in the first two elements of the observation

        # Check if the agent revisits a position
        if taxi_pos in self.visited_positions:
            # Apply a penalty if the taxi revisits a position
            reward -= self.penalty
        else:
            # Mark the current position as visited
            self.visited_positions.add(taxi_pos)

        # If the agent successfully picks up or drops off a passenger, we don't penalize
        if done and 'pickup' in info and info['pickup']:
            self.visited_positions.clear()  # Reset visited positions after task completion

        return obs, reward, done, _, info
    



class CustomActionWrapper(ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        # Extend the action space to include diagonal movements
        self.action_space = gymnasium.spaces.Discrete(env.action_space.n + 4)
    
    def step(self, action):
        grid_size = 5

        # get the taxi position
        taxi_row, taxi_col, pass_idx, dest_idx = self.env.unwrapped.decode(self.state)


        # 6: Move South-East --- Increase row, increase column
        # 7: Move South-West --- Increase row, decrease column
        # 8: Move North-East --- Decrease row, increase column
        # 9: Move North-West --- Decrease row, decrease column

        if action == 6:
            taxi_row = min(taxi_row + 1, grid_size - 1)
            taxi_col = min(taxi_col + 1, grid_size - 1)

        elif action == 7:
            taxi_row = min(taxi_row + 1, grid_size - 1)
            taxi_col = max(taxi_col - 1, 0)
        elif action == 8:
            taxi_row = max(taxi_row - 1, 0)
            taxi_col = min(taxi_col + 1, grid_size - 1)
        elif action == 9:
            taxi_row = max(taxi_row - 1, 0)
            taxi_col = max(taxi_col - 1, 0)
        else:
            return action

        # Update the environment state manually for diagonal actions
        self.env.unwrapped.s = (taxi_row, taxi_col, pass_idx, dest_idx)
        obs, reward, done, truncated, info = self.env.unwrapped.step(6)  # No-op to get updated info
        return obs, reward, done, truncated, info
        
    




class CustomActionWrapper(ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.action_space = gym.spaces.Discrete(env.action_space.n + 4)

    def action(self, action):
        """Transform the custom action into a base action if necessary."""
        if action in [6, 7, 8, 9]:
            # Diagonal actions are handled in the `step` method.
            return None  # Indicate custom handling
        return action  # Pass through for base actions

    # def step(self, action):
    #     grid_size = 5

    #     # Decode the state
    #     encoded_state = self.env.unwrapped.s
    #     taxi_row, taxi_col, pass_idx, dest_idx = self.env.unwrapped.decode(encoded_state)

    #     # Handle diagonal actions
    #     if action == 6:  # Move South-East
    #         new_row = min(taxi_row + 1, grid_size - 1)
    #         new_col = min(taxi_col + 1, grid_size - 1)
    #     elif action == 7:  # Move South-West
    #         new_row = min(taxi_row + 1, grid_size - 1)
    #         new_col = max(taxi_col - 1, 0)
    #     elif action == 8:  # Move North-East
    #         new_row = max(taxi_row - 1, 0)
    #         new_col = min(taxi_col + 1, grid_size - 1)
    #     elif action == 9:  # Move North-West
    #         new_row = max(taxi_row - 1, 0)
    #         new_col = max(taxi_col - 1, 0)
    #     else:
    #         # Pass through base actions
    #         return super().step(action)

    #     # Validate the move (check for walls or invalid spaces)
    #     if self.env.desc[new_row, new_col] != b' ':  # Assume `b' '` indicates valid space
    #         # Invalid move, no state change
    #         reward = -1  # Same as base environment for invalid moves
    #         done = False
    #         truncated = False
    #         obs = self.env.unwrapped.s
    #     else:
    #         # Valid move, update the state
    #         self.env.unwrapped.s = self.env.unwrapped.encode(new_row, new_col, pass_idx, dest_idx)
    #         reward = -1  # Same as base environment
    #         done = self.env.unwrapped.s == self.env.unwrapped.encode(
    #             dest_idx // grid_size, dest_idx % grid_size, pass_idx, dest_idx
    #         )
    #         truncated = False
    #         obs = self.env.unwrapped.s

    #     # Return updated state
    #     info = {'diagonal_action': action in [6, 7, 8, 9]}
    #     return obs, reward, done, truncated, info


    def step(self, action):
        grid_size = 5

        # Get the taxi position
        encoded_state = self.env.unwrapped.s
        taxi_row, taxi_col, pass_idx, dest_idx = self.env.unwrapped.decode(encoded_state)

        # Handle diagonal actions
        if action == 6:  # Move South-East
            new_row = min(taxi_row + 1, grid_size - 1)
            new_col = min(taxi_col + 1, grid_size - 1)
        elif action == 7:  # Move South-West
            new_row = min(taxi_row + 1, grid_size - 1)
            new_col = max(taxi_col - 1, 0)
        elif action == 8:  # Move North-East
            new_row = max(taxi_row - 1, 0)
            new_col = min(taxi_col + 1, grid_size - 1)
        elif action == 9:  # Move North-West
            new_row = max(taxi_row - 1, 0)
            new_col = max(taxi_col - 1, 0)
        else:
            # Pass through base actions to the original step method
            return super().step(action)

        # Validate the move (check for walls or invalid spaces)
        if self.env.unwrapped.desc[new_row, new_col] != b' ':  # Assume `b' '` indicates valid space
            # Invalid move, no state change
            reward = -1  # Same as base environment for invalid moves
            done = False
            obs = self.env.unwrapped.s
        else:
            # Update the state manually for diagonal actions
            self.env.unwrapped.s = self.env.unwrapped.encode(new_row, new_col, pass_idx, dest_idx)

            # Compute reward manually
            reward = -1  # Default reward for non-goal moves

            # Check if the new state is terminal
            done = self.env.unwrapped.s == self.env.unwrapped.encode(
                dest_idx // grid_size, dest_idx % grid_size, pass_idx, dest_idx
            )

            # Get updated observation
            obs = self.env.unwrapped.s

        truncated = False  # Taxi environment doesn't use truncation

        # Return updated information
        return obs, reward, done, truncated, {}























import gymnasium as gym
# Create the environment with DummyVecEnv
env = gym.make("Taxi-v3")

# Wrap the environment with the RevisitPenaltyWrapper
env = CustomActionWrapper(env)
extended_callback = ExtendedCallback()

# Create your model (for example, PPO)
from stable_baselines3 import PPO
model_ppo = PPO("MlpPolicy", env, verbose=0)

# Train the model with the wrapper applied
model_ppo.learn(total_timesteps=60000000, callback=extended_callback)

# Save the model
model_ppo.save("ppo_custom_action_taxi")
