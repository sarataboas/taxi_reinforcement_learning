import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.toy_text.taxi import TaxiEnv
import numpy as np


class CustomTaxiEnv(TaxiEnv):

    metadata = {'render.modes': ['human', 'rgb_array', 'ansi'], 'render.fps': 4}

    def __init__(self, grid_size= (10,10), render_mode = None, obstacles = None, possible_locations = None):
        super().__init__()

        #changes 
        self.grid_size = grid_size
        self.grid_rows = grid_size[0]
        self.grid_cols = grid_size[1]
        self.possible_locations = possible_locations
        self.number_destinations = len(self.possible_locations)
        assert self.number_destinations > 0, "number_destinations must be positive."

        self.obstacles = obstacles if obstacles else []
        self.possible_locations = possible_locations if possible_locations else []
        self.action_space = gym.spaces.Discrete(6) # action space - 6 actions
        self.observation_space = gym.spaces.Discrete(self.grid_rows * self.grid_cols * (self.grid_rows * self.grid_cols + 1) * self.number_destinations) # observation space
       
        # pygame 
        self.window_size = 512 # size of the window pygame 
        
        assert render_mode is None or render_mode in self.metadata['render.modes']
        self.render_mode = render_mode
        '''
           If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        '''
        self.window = None
        self.clock = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Random initial positions
        taxi_row = self.np_random.integers(0, self.grid_rows)
        taxi_col = self.np_random.integers(0, self.grid_cols)
        passenger_location = self.np_random.integers(0, self.grid_rows * self.grid_cols)
        destination = self.np_random.integers(0, self.number_destinations)

        # Encode initial state
        self.state = self.encode_state(taxi_row, taxi_col, passenger_location, destination)
        info = self.get_info(taxi_row, taxi_col, passenger_location, destination)

        if self.render_mode == "human":
            self.render()

        return self.state, info

        
    def get_obs(self, taxi_row, taxi_col, passenger_location, destination):
        # observations is an integer that encodes the corresponding state 
        # calculated by : ((taxi_row * 5 + taxi_col) * 5) + passenger_location) * 4 + destination --> original environment
        # expression for this customization :
        self.obs = ((taxi_row * self.grid_cols + taxi_col) * (self.grid_rows * self.grid_cols + 1) + passenger_location) * self.number_destinations + destination
        return self.obs

    def get_info(self, taxi_row, taxi_col, passenger_location, destination):
        """
        Provide additional information about the current state and valid actions.

        - p: Transition probability (always 1.0 in deterministic environments like Taxi).
        - action_mask: A numpy array indicating which actions are valid.
        """
        # Transition probability
        p = 1.0  # This environment is deterministic

        # Action mask: Check which actions are valid
        action_mask = np.zeros(6, dtype=np.int32)

        # Up action: Can move up if not at the top row
        action_mask[0] = int(taxi_row > 0)

        # Down action: Can move down if not at the bottom row
        action_mask[1] = int(taxi_row < self.grid_rows - 1)

        # Left action: Can move left if not at the leftmost column
        action_mask[2] = int(taxi_col > 0)

          # Right action: Can move right if not at the rightmost column
        action_mask[3] = int(taxi_col < self.grid_cols - 1)

        # Pickup action: Valid if passenger is at the taxi's location
        action_mask[4] = int(passenger_location == (taxi_row * self.grid_cols + taxi_col))

        # Dropoff action: Valid if passenger is in the taxi and at the destination
        action_mask[5] = int(passenger_location == self.grid_rows * self.grid_cols and
                             destination == (taxi_row * self.grid_cols + taxi_col))

        return {
            "p": p,
            "action_mask": action_mask
        }

    def build_transitional_model(self):
        self.P = {}
        for state in range(self.observation_space.n):
            self.P[state] = {action: [] for action in range(self.action_space.n)}
            # Decode state into taxi_row, taxi_col, passenger_location, destination
            taxi_row, taxi_col, passenger_location, destination = self.decode_state(state)
            
            for action in range(self.action_space.n):
                new_state, reward, done = self.get_transition(state, action)
                self.P[state][action].append((1.0, new_state, reward, done))  # Probabilities are deterministic (1.0)

    def get_transition(self, state, action):
        # Decode the current state
        taxi_row, taxi_col, passenger_location, destination = self.decode_state(state)

        # Default new values
        new_taxi_row, new_taxi_col = taxi_row, taxi_col
        new_passenger_location, new_destination = passenger_location, destination
        reward, done = -1, False  # Default reward for each action is -1

        # Actions: 0-South, 1-North, 2-East, 3-West, 4-Pickup, 5-Dropoff
        if action == 0:  # Move south
            if taxi_row < self.grid_size[0] - 1 and (taxi_row + 1, taxi_col) not in self.obstacles:
                new_taxi_row += 1
            else:
                reward -= 1  # Penalty for hitting a boundary/obstacle
        elif action == 1:  # Move north
            if taxi_row > 0 and (taxi_row - 1, taxi_col) not in self.obstacles:
                new_taxi_row -= 1
            else:
                reward -= 1
        elif action == 2:  # Move east
            if taxi_col < self.grid_size[1] - 1 and (taxi_row, taxi_col + 1) not in self.obstacles:
                new_taxi_col += 1
            else:
                reward -= 1
        elif action == 3:  # Move west
            if taxi_col > 0 and (taxi_row, taxi_col - 1) not in self.obstacles:
                new_taxi_col -= 1
            else:
                reward -= 1
        elif action == 4:  # Pickup
            # Check if taxi is at a passenger location and passenger is not already in-taxi
            if passenger_location < len(self.possible_locations) and \
            (taxi_row, taxi_col) == self.possible_locations[passenger_location]:
                new_passenger_location = len(self.possible_locations)  # Passenger is now "in the taxi"
                reward = 10  # Reward for successful pickup
            else:
                reward = -10  # Penalty for illegal pickup
        elif action == 5:  # Dropoff
            # Check if passenger is in the taxi and taxi is at the destination
            if passenger_location == len(self.possible_locations) and \
            (taxi_row, taxi_col) == self.possible_locations[destination]:
                reward = 20  # Reward for successful dropoff
                done = True  # Episode ends
            else:
                reward = -10  # Penalty for illegal dropoff
        else:
            raise ValueError("Invalid action")

        # Encode the new state
        new_state = self.encode_state(new_taxi_row, new_taxi_col, new_passenger_location, new_destination)
        return new_state, reward, done


    def encode_state(self, taxi_row, taxi_col, passenger_location, destination):
        return ((taxi_row * self.grid_cols + taxi_col) * 
                (self.grid_rows * self.grid_cols + 1) + 
                passenger_location) * self.number_destinations + destination

    def decode_state(self, state):
        destination = state % self.number_destinations
        state //= self.number_destinations
        passenger_location = state % (self.grid_rows * self.grid_cols + 1)
        state //= (self.grid_rows * self.grid_cols + 1)
        taxi_col = state % self.grid_cols
        taxi_row = state // self.grid_cols
        return taxi_row, taxi_col, passenger_location, destination
    
    def step(self, action):
        new_state, reward, done = self.get_transition(self.state, action)
        self.state = new_state
        info = self.get_info(*self.decode_state(new_state))
        return self.state, reward, done, False, info

    # optional render method to add pygame
    def render(self):
        print("Rendering")
        pass 



# # ------- Test the environment -------
# def test_env():
#     env = CustomTaxiEnv(grid_size=(10,10), number_destinations=4)

#     state, info = env.reset()
#     print(f"Initial state: {state}")
#     print(f"Initial info: {info}")

#     # run a few steps in the environment 
#     num_steps = 10
#     total_reward = 0

#     for step in range(num_steps):
#         # Sample a ramdom action
#         action = env.action_space.sample()
#         print(f"Step {step + 1}: Taking action {action}")

#         # take a step in the environment
#         state, reward, done, info = env.step(action)

#         print(f"New state: {state}")
#         print(f"Reward: {reward}")
#         print(f"Done: {done}")
#         print(f"Info: {info}\n")

#         total_reward += reward

#         if done:
#             print("Episode finished.")
#             break

#     print(f"\nTotal reward after {step + 1} steps: {total_reward}")



# test_env()

