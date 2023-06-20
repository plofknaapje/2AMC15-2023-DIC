"""Random Agent.

This is an agent that takes a random action from the available action space.
"""
import random
from random import randint
import numpy as np
import math
from agents import BaseAgent
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque


class DQN(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(num_inputs, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class DQNAgent(BaseAgent):
    def __init__(self, agent_number: int, gamma: float, grid_size: int, epsilon=0.4, alpha=0.001):
        """
        Set agent parameters.

        Args:
            agent_number: The index of the agent in the environment.
            gamma: loss rate.
            theta: minimal change.
            epsilon: epsilon greedy
            alpha: learning rate
        """
        super().__init__(agent_number)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay_steps = 1000000000
        self.epsilon_min = 0.02
        self.alpha = alpha
        self.num_actions = 4

        self.grid_size = grid_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.Q_network = DQN(self.grid_size, self.num_actions).to(self.device)
        self.target_network = DQN(self.grid_size, self.num_actions).to(self.device)
        self.target_network.load_state_dict(self.Q_network.state_dict())
        self.target_network.eval()

        self.dirtGrid = np.zeros(4)

        self.state = [0, 0, 0]
        self.new_state = [0, 0, 0]

        self.optimizer = optim.Adam(self.Q_network.parameters(), lr=self.alpha)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.buffer_max_size = 50000
        self.buffer_min_size = 1000
        self.replay_buffer = deque(maxlen=self.buffer_max_size)
        self.batch_size = 64
        self.target_update_freq = 1000

        self.reward_buffer = deque(maxlen=1000)
        self.step = 0

    def process_reward(self, pos: np.ndarray, reward: float, action: int, done: int):
        # # update the Q function
        # self.Q[self.state[0], self.state[1], self.state[2], action] += \
        #     self.alpha_decay * (reward + self.gamma * np.max(
        #         self.Q[self.new_state[0], self.new_state[1], self.new_state[2], :]) -
        #                         self.Q[self.state[0], self.state[1], self.state[2], action])

        self.new_state = torch.tensor(pos.flatten(), dtype=torch.float32).unsqueeze(0).to(self.device)
        transition = (self.state, action, reward, done, self.new_state)
        self.replay_buffer.append(transition)
        self.state = self.new_state

        self.reward_buffer.append(reward)

        return 0

    def train(self):
        if len(self.replay_buffer) > self.batch_size:
            batch = random.sample(self.replay_buffer, self.batch_size)
            state_batch, action_batch, reward_batch, done_batch, new_state_batch = zip(*batch)

            state_batch = torch.stack(state_batch).to(self.device)
            action_batch = torch.tensor(action_batch).to(self.device)
            reward_batch = torch.tensor(reward_batch).unsqueeze(1).to(self.device)
            done_batch = torch.tensor(done_batch).unsqueeze(1).to(self.device)
            new_state_batch = torch.stack(new_state_batch).to(self.device)


            Q_values = self.Q_network(state_batch)
            max_new_Q_values = self.target_network(new_state_batch)
            max_new_Q_values = torch.max(max_new_Q_values, dim=2)[0]

            expected_Q_values = reward_batch + self.gamma * max_new_Q_values * (1 - done_batch)

            action_batch = action_batch.unsqueeze(1).unsqueeze(1)

            action_q_values = torch.gather(input=Q_values, dim=2, index=action_batch)
            action_q_values = action_q_values.squeeze(1)

            loss = F.smooth_l1_loss(action_q_values, expected_Q_values)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.step = self.step + 1
            if self.step % self.target_update_freq == 0:
                self.target_network.load_state_dict(self.Q_network.state_dict())
                print(np.mean(self.reward_buffer))
                print(self.step)
                print(self.epsilon)

    def take_action(self, pos: np.ndarray, info: None | dict):

        self.train()

        self.epsilon = max(self.epsilon * (1 - (self.step / self.epsilon_decay_steps)), self.epsilon_min)

        self.state = torch.tensor(pos.flatten(), dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            q_values = self.Q_network(self.state)

        if np.random.uniform(0, 1) < self.epsilon:
            action = random.randint(0, self.num_actions - 1)
        else:
            action = q_values.argmax(dim=1).item()

        return action

    def take_action_eval(self, pos: np.ndarray, info: None | dict):
        self.state = torch.tensor(pos.flatten(), dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            q_values = self.Q_network(self.state)

        print(q_values)
        action = q_values.argmax(dim=1).item()

        return action





