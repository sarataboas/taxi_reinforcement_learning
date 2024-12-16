import gymnasium as gym
from gymnasium.envs.toy_text.taxi import TaxiEnv
import numpy as np

# ------------Test the Taxi-v3 original environment -------------------

env = gym.make('Taxi-v3', render_mode = 'rgb_array')

obs, info = env.reset()

print(f"Initial state: ")
env.render()

state_desc = env.unwrapped.decode(obs)

taxi_row, taxi_col, passenger_location, destination = state_desc

print(f"obs: {obs}")
print(f"info: ", info)
print(f"Taxi location: row {taxi_row}, col {taxi_col}")
print(f"Passenger location: {passenger_location}")
print(f"Destination: {destination}")
# -------------------------------



# # -------------- Environment Customization ------------------
# '''
# ####  Action space: 
#         0: Move south (down)
#         1: Move north (up)
#         2: Move east (right)
#         3: Move west (left)
#         4: Pickup passenger
#         5: Drop off passenger

# ####  Observation space: there are 500 discrete states --> 25 taxi positions, 5 possible locations of the passenger (including the case when the passenger is in the taxi), 
#                                                                                    and 4 destination locations.
#         - passenger locations :     0: Red
#                                     1: Green
#                                     2: Yellow
#                                     3: Blue
#                                     4: In taxi 

#         - destination locations :   0: Red
#                                     1: Green
#                                     2: Yellow
#                                     3: Blue

# ####  Rewards Customization:
#         - For each action taken, the agent receives a reward of -1
#         - Illegal pickup: -10
#         - Successful pickup: +10
#         - Illegal dropoff: -10
#         - Successful dropoff: +20
#         - Taxi in obstacle: -5
#         - Bonus for efficient routes taken ---- FALTA
#         - Passenger preferences ---- FALTA

# #### Space Customization:
#         - increasing the grid size
#         - adding obstacles
# '''

# MAP = [
#     "+---------+",
#     "|R: | : :G|",
#     "| : | : : |",
#     "| : : : : |",
#     "| | : | : |",
#     "|Y| : |B: |",
#     "+---------+",
# ]
# WINDOW_SIZE = (550, 350)



# class CustomTaxiEnv(TaxiEnv):

#     metadata = {
#         "render_modes": ["human", "ansi", "rgb_array"],
#         "render_fps": 4, # the environment will be rendered at 4 frames per second
#     }


#     def __init__(self, grid_size=(10,10), obstacles=None):
#         super().__init__()
#         self.grid_size = grid_size
#         self.possible_locations = [
#             (0, 0),  # Red
#             (0, grid_size[1] - 1),  # Green
#             (grid_size[0] - 1, 0),  # Yellow
#             (grid_size[0] - 1, grid_size[1] - 1),  # Blue
#         ]
#         self.action_space = gym.spaces.Discrete(6) # action space - 6 actions
#         self.observation_space = gym.spaces.Discrete(self.grid_size[0] * self.grid_size[1] * len(self.possible_locations) * len(self.possible_locations))
#         self.obstacles = obstacles if obstacles else []
#         self.passenger_location = None
#         self.destination = None
#         self.taxi_position = None

        
#         # Ensure valid initial state
#         self._initialize_state()

#     def _initialize_state(self):
#         self.taxi_position = (
#         np.random.randint(0, self.grid_size[0]),
#         np.random.randint(0, self.grid_size[1]))

#         print(f"Initialized Taxi Position: {self.taxi_position}")  # Debugging

#         # Ensure valid initial passenger location and destination
#         self.passenger_location = np.random.choice(len(self.possible_locations))  # Valid indices: 0-4
#         self.destination = np.random.choice(
#             [i for i in range(len(self.possible_locations)) if i != self.passenger_location])
        


#     def reset(self):
#         self._initialize_state()
#         obs = self.encode(self.taxi_position[0], self.taxi_position[1], )
#         self.s = obs  # Update `self.s` to the current state
#         info = {}
#         return obs, info


#     def step(self, action):
#         reward = 0
#         taxi_row, taxi_col = self.taxi_position
#         done = False
        
#         # Logic for movement and actions
#         if action == 0:  # Move south
#             if taxi_row < self.grid_size[0] - 1 and (taxi_row + 1, taxi_col) not in self.obstacles:
#                 self.taxi_position = (taxi_row + 1, taxi_col)
#             else:
#                 reward -= 1  # Penalty for hitting boundary/obstacle
#         elif action == 1:  # Move north
#             if taxi_row > 0 and (taxi_row - 1, taxi_col) not in self.obstacles:
#                 self.taxi_position = (taxi_row - 1, taxi_col)
#             else:
#                 reward -= 1
#         elif action == 2:  # Move east
#             if taxi_col < self.grid_size[1] - 1 and (taxi_row, taxi_col + 1) not in self.obstacles:
#                 self.taxi_position = (taxi_row, taxi_col + 1)
#             else:
#                 reward -= 1
#         elif action == 3:  # Move west
#             if taxi_col > 0 and (taxi_row, taxi_col - 1) not in self.obstacles:
#                 self.taxi_position = (taxi_row, taxi_col - 1)
#             else:
#                 reward -= 1
#         elif action == 4:  # Pickup
#             if (taxi_row, taxi_col) == self.possible_locations[self.passenger_location]:
#                 self.passenger_location = len(self.possible_locations)  # "In taxi"
#                 reward += 10
#             else:
#                 reward -= 10
#         elif action == 5:  # Dropoff
#             if self.passenger_location == len(self.possible_locations) and (taxi_row, taxi_col) == self.possible_locations[self.destination]:
#                 reward += 20
#                 done = True  # Episode ends upon successful dropoff
#             else:
#                 reward -= 10
#         else:
#             raise ValueError("Invalid action")

#         # Generate the next state
#         # obs = self.get_state()
#         self.s = obs  # Update `self.s` to the current state
#         return obs, reward, done, info



# env = CustomTaxiEnv(grid_size=(10, 10), obstacles=[(1, 1), (2, 2)])

# info = env.reset()

# for step in range(10000):
#     action = env.action_space.sample()  # Random action
#     obs, reward, done, info = env.step(action)
#     print(f"Step: {step}, Action: {action}, Observation: {obs}, Reward: {reward}, Done: {done}")
#     env._render_gui('human')

#     if done:
#         print("Episode finished!")
#         break

# env.close()
