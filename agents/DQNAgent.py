"""Random Agent.

This is an agent that takes a random action from the available action space.
"""
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
from pathlib import Path

from agents import BaseAgent


class DQN(nn.Module):
    def __init__(self, num_inputs: int, num_actions: int):
        """
        DQN. Creates a new Deep Q-learning Network.

        Args:
            num_inputs (int): size of the input data.
            num_actions (int): number of possible actions.
        """        
        super(DQN, self).__init__()
        # neural network with 5 layers.
        self.fc1 = nn.Linear(num_inputs, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 32)
        self.fc5 = nn.Linear(32, num_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the DQN.

        Args:
            x (torch.Tensor): input data of the network.

        Returns:
            torch.Tensor: output of the network.
        """
        # Every hidden layer passes through ReLU activation.        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x


class DQNAgent(BaseAgent):
    def __init__(self, agent_number: int, gamma: float, grid_size: int, epsilon=0.4, alpha=0.001, 
                 target_update_freq=1000, verbose=False):
        """
        Set agent parameters.

        Args:
            agent_number (int): The index of the agent in the environment.
            gamma (float): loss rate.
            grid_size (int): number of spaces in the grid.
            epsilon (float): epsilon greedy. Defaults to 0.4.
            alpha (float): learning rate. Defaults to 0.001.
            target_update_freq (int): per how many steps should the target network be updated. Defaults to 1000.
            verbose (bool): report on internal operations. Defaults to False.
        """

        super().__init__(agent_number)

        # Agent hyperparameters.
        self.gamma = gamma
        self.epsilon_start = epsilon
        self.epsilon = epsilon
        self.epsilon_min = 0.02
        self.alpha = alpha

        # Agent parameters
        self.grid_size = grid_size + 30
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.buffer_max_size = 50000
        self.buffer_min_size = 1000
        self.replay_buffer = deque(maxlen=self.buffer_max_size)
        self.batch_size = 64
        self.target_update_freq = target_update_freq
        self.reward_buffer = deque(maxlen=1000)
        self.num_actions = 4
        self.verbose = verbose

        # Agent information
        self.Q_network = DQN(self.grid_size, self.num_actions).to(self.device)
        self.target_network = DQN(self.grid_size, self.num_actions).to(self.device)
        self.target_network.load_state_dict(self.Q_network.state_dict())
        self.target_network.eval()
        self.step = 0
        self.state = [0, 0, 0]
        self.new_state = [0, 0, 0]

        # Agent learning infrastructure
        self.optimizer = optim.Adam(self.Q_network.parameters(), lr=self.alpha)


    def process_reward(self, pos: np.ndarray, reward: float, action: int, done: int):
        """
        Process the rewards of the past actions.

        Args:
            pos (np.ndarray): current position of the agent.
            reward (float): reward of the action(s).
            action (int): action chosen.
            done (int): action was actually taken.
        """        
        self.new_state = torch.tensor(pos.flatten(), dtype=torch.float32).unsqueeze(0).to(self.device)
        transition = (self.state, action, reward, done, self.new_state)
        self.replay_buffer.append(transition)
        self.state = self.new_state
        self.reward_buffer.append(reward)

    def train(self):
        """
        Trains the complete agent based on the defined hyperparameters and learning infrastructure.
        """        
        if len(self.replay_buffer) > self.batch_size:

            # Data preparation
            batch = random.sample(self.replay_buffer, self.batch_size)
            state_batch, action_batch, reward_batch, done_batch, new_state_batch = zip(*batch)

            state_batch = torch.stack(state_batch).to(self.device)
            action_batch = torch.tensor(action_batch).to(self.device)
            reward_batch = torch.tensor(reward_batch).unsqueeze(1).to(self.device)
            done_batch = torch.tensor(done_batch).unsqueeze(1).to(self.device)
            new_state_batch = torch.stack(new_state_batch).to(self.device)

            # Push data through the network
            Q_values = self.Q_network(state_batch)
            max_new_Q_values = self.target_network(new_state_batch)
            max_new_Q_values = torch.max(max_new_Q_values, dim=2)[0]
            expected_Q_values = reward_batch + self.gamma * max_new_Q_values * (1 - done_batch)

            action_batch = action_batch.unsqueeze(1).unsqueeze(1)
            action_q_values = torch.gather(input=Q_values, dim=2, index=action_batch)
            action_q_values = action_q_values.squeeze(1)

            # Process results
            loss = F.smooth_l1_loss(action_q_values, expected_Q_values)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.step = self.step + 1
            # Target network update step
            if self.step % self.target_update_freq == 0 and self.verbose:
                self.target_network.load_state_dict(self.Q_network.state_dict())
                print(np.mean(self.reward_buffer))
                print(self.step)
                print(self.epsilon)

    def take_action(self, pos: np.ndarray, info: None | dict) -> int:
        """
        Take an action in the training phase.

        Args:
            pos (np.ndarray): current position.
            info (None | dict): environment info.

        Returns:
            int: action to be taken.
        """        
        self.train()
        # Epsilon is decreased over time
        self.epsilon = max(self.epsilon_start * (1 - info['iteration']), self.epsilon_min)
        self.state = torch.tensor(pos.flatten(), dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            q_values = self.Q_network(self.state)

        # Choose random actions sometimes.
        if np.random.uniform(0, 1) < self.epsilon:
            action = random.randint(0, self.num_actions - 1)
        else:
            action = q_values.argmax(dim=1).item()

        return action

    def take_action_eval(self, pos: np.ndarray, info: None | dict) -> int:
        """
        Take an action in the evaluation phase

        Args:
            pos (np.ndarray): current position.
            info (None | dict): environment info.

        Returns:
            int: action to be taken.
        """        
        self.state = torch.tensor(pos.flatten(), dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            q_values = self.Q_network(self.state)

        if self.verbose:
            print(q_values)
        action = q_values.argmax(dim=1).item()

        return action

    def save_model(self, dynamic=True):
        """
        Save the trained Q network to the DQN_models folder with its settings.
        """
        if dynamic:        
            torch.save(self.Q_network.state_dict(), 
                       Path(f"DQN_models/model_updaterate{self.target_update_freq}_gamma{self.gamma}_alpha{self.alpha}_dynamic.pt"))
        else:
            torch.save(self.Q_network.state_dict(), 
                       Path(f"DQN_models/model_updaterate{self.target_update_freq}_gamma{self.gamma}_alpha{self.alpha}_static.pt"))    

    def load_model(self, model_path: str | Path):
        """
        Load a trained Q network from the supplied path

        Args:
            model_path (str | Path): path to the saved model.
        """
        self.Q_network.load_state_dict(torch.load(model_path, map_location=self.device))

    def __str__(self):
        return f"DQNAgent({self.agent_number}, {self.target_update_freq}, {self.gamma}, {self.alpha})"
    