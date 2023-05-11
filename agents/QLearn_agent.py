"""Random Agent.

This is an agent that takes a random action from the available action space.
"""
from random import randint
import numpy as np

from agents import BaseAgent


class QLearnAgent(BaseAgent):
    def __init__(self, agent_number: int, gamma: float, theta=0.001, epsilon=0.05, alpha=0.001):
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
        self.theta = theta
        self.epsilon = epsilon
        self.alpha = alpha
        self.Q = np.zeros([2,2])

    def process_reward(self, observation: np.ndarray, reward: float):
        pass

    def take_action(self, observation: np.ndarray, info: None | dict) -> int:
        # take action according to epsilon greedy
        if np.random.uniform(0, 1) < self.epsilon:
            action = np.random.randint(0, 4)
        else:
            action = np.argmax(self.Q[state,])

        # compute reward
        reward = 1

        # get new state
        new_state = 1

        # update the Q function
        Q[state, action] += self.alpha * (reward + self.gamma * np.max(Q[new_state,]) - Q[state, action])

