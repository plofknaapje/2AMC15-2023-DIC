from collections import defaultdict
import numpy as np
from agents.base_agent import BaseAgent

from dataclasses import dataclass
from typing import Tuple

# @dataclass(frozen=True)
class CustomState:
    """
    Custom state representation.

    Attributes:
        grid (np.ndarray): The grid representing the state.
        pos (Tuple[int, int]): The position in the grid.
        is_terminal (bool): Flag indicating if the state is terminal.
    """
    def __init__(self, grid: np.ndarray, pos: Tuple[int, int], is_terminal: bool = False):
        self.grid = grid
        self.pos = pos
        self.is_terminal = is_terminal

    def __hash__(self) -> int:
        """Ensure state is hashable to make it easier to work with"""
        return hash((np.array2string(self.grid), self.pos))

    def __eq__(self, other: object) -> bool:
        """Check equality between two CustomState objects"""
        self_ = (self.pos, np.array2string(self.grid))
        other_ = (other.pos, np.array2string(other.grid))
        return self_ == other_

    def __lt__(self, other: "CustomState"):
        """Ensure sortability of CustomState objects"""
        return hash(self) < hash(other)

    def __str__(self):
        """String representation of the CustomState object"""
        return np.array2string(self.grid) + ": " + str(self.pos)

class QLearnAgent(BaseAgent):
    """
    TD Agent. This is an agent that implements the Q-learning Temporal Difference algorithm.
    """
    def __init__(self, agent_number, allowed_actions, training, gamma=0.6, alpha=0.25, epsilon=0.05):
        """
        Initialize the Q-learning agent.

        Args:
            agent_number (int): The agent's number.
            allowed_actions (list): List of allowed actions.
            training (bool): Flag indicating if the agent is in training mode.
            gamma (float): Discount factor for future rewards.
            alpha (float): Learning rate of the agent.
            epsilon (float): Probability of taking a random action for exploration.
        """
        super().__init__(agent_number)
        self._allowed_actions = allowed_actions
        self.Q_table = defaultdict(lambda: np.zeros(len(self._allowed_actions)))
        self.epsilon = epsilon
        self.alpha = alpha
        self.training = training
        self.gamma = gamma
        self.last_action = None
        self.last_state = None

    def process_reward(self, observation: np.ndarray, info: None | dict, reward: float, terminated: bool):
        """
        Process the reward and update the Q-table according to SARS.

        Args:
            observation (np.ndarray): The observation returned by the environment.
            info (None or dict): Additional information for the agent.
            reward (float): The reward returned by the environment.
            terminated (bool): Flag indicating if this is the last reward of the episode.

        Returns:
            float: The calculated delta value.
        """
        agent_position = info["agent_pos"]
        new_state = str(CustomState(observation, agent_position[0]))
        if terminated:
            delta = self.alpha * (reward - self.Q_table[self.last_state][self.last_action])
            self.Q_table[self.last_state][self.last_action] += delta
        else:
            delta = self.alpha * (reward + self.gamma * max(self.Q_table[new_state]) - self.Q_table[self.last_state][self.last_action])
            self.Q_table[self.last_state][self.last_action] += delta
        return delta

    def take_action(self, observation: np.ndarray, info: None | dict) -> int:
        """
        Choose the best action to take from the given state, using the Q-table.

        During the training phase, the agent makes a random move with a probability of epsilon to support exploration.

        Args:
            observation (np.ndarray): The observation returned by the environment.
            info (None or dict): Additional information for the agent.

        Returns:
            int: The chosen action.
        """
        agent_position = info["agent_pos"]
        state = str(CustomState(observation, agent_position[0]))

        if self.training:
            if np.random.uniform() <= self.epsilon:
                # Take a random action
                action = np.random.choice(self._allowed_actions)
                self.last_action = action
                self.last_state = state
                return action
            else:
                # Take the greedy action
                action = np.argmax(self.Q_table[state])
                self.last_action = action
                self.last_state = state
                return action
        else:
            return np.argmax(self.Q_table[state])
