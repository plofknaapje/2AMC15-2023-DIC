"""Monte Carlo Agent.

Uses Monte Carlo estimation to determine the values and optimal policy.
"""

import numpy as np

from agents import BaseAgent

from itertools import chain, combinations
from random import choice


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)  # allows duplicate elements
    return [set for set in
            chain.from_iterable(combinations(s, r) for r in range(len(s)+1))]


class MCAgent(BaseAgent):
    def __init__(self, agent_number: int, gamma: float, obs: np.ndarray):
        """
        Sets agent parameters.

        Args:
            agent_number (int): The index of the agent in the environment.
            gamma (float): loss rate.
            obs (np.ndarray): environment observation.
        """
        super().__init__(agent_number)
        self.gamma = gamma

        cols, rows = obs.shape

        self.spaces = [
            (i, j) for i in range(cols) for j in range(rows)
            if obs[i, j] in (0, 3, 4)
        ]

        self.dirt_spaces = [
            space for space in self.spaces
            if obs[space] == 3
        ]

        dirt_configs = powerset(self.dirt_spaces)

        self.states = [
            (space, dirt)
            for space in self.spaces
            for dirt in dirt_configs
        ]

        self.returns = {(state, action): [] for state in self.states
                        for action in [0, 1, 2, 3]}

        self.q = {(state, action): 0 for state in self.states
                  for action in [0, 1, 2, 3]}

        self.pi = {state: -1 for state in self.states}

        # Agent state
        self.episode = []
        self.rewards = []
        self.dirt_left = self.dirt_spaces.copy()
        self.prev_reward = 0

    def process_reward(self, observation: np.ndarray, reward: float):
        """
        Process the rewards from the epoch. The discounted reward is calculated for each action.
        Then, these rewards are added to the rewards of each state-action pair.

        Args:
            observation (np.ndarray): current observation.
            reward (float): reward value.
        """

        if sum(self.rewards) > 0:
            discounted_rewards = []
            for reward in reversed(self.rewards):
                if len(discounted_rewards) > 0:
                    discounted_rewards.append(discounted_rewards[-1]**self.gamma + reward)
                else:
                    discounted_rewards.append(reward)
            discounted_rewards.reverse()

            episode_states = set()
            for i, (state, action) in enumerate(self.episode):
                self.returns[state, action].append(discounted_rewards[i])
                self.q[state, action] = np.mean(self.returns[state, action])
                episode_states.add(state)

            for state in episode_states:
                self.pi[state] = self.best_action(state)

        self.reset_agent_state()

    def reset_agent_state(self):
        """
        Reset the internal record keeping of the agent for a new epoch.
        """
        self.episode = []
        self.rewards = []
        self.prev_reward = 0
        self.dirt_left = self.dirt_spaces.copy()

    def take_action(self, observation: np.ndarray, info: None | dict) -> int:
        """
        Default method which decides an action based on the environment.

        Args:
            observation: current environment.
            info: situation in current environment.

        Returns:
            int: action to be taken.
        """
        agent_space = info["agent_pos"][self.agent_number]
        self.dirt_left = [space for space in self.spaces if observation[space] == 3]

        self.old_pos = agent_space

        state = (agent_space, tuple(self.dirt_left))

        action = self.pi[state]
        if action == -1:
            action = self.random_action(state)

        self.episode.append((state, action))

        return action

    def random_action(self, state: tuple) -> int:
        """
        Generate a random action.

        Args:
            state (tuple): current state.

        Returns:
            tuple: proposed action.
        """
        options = []

        for action in [0, 1, 2, 3]:
            new_space = self.action_result(state, action)
            if new_space in self.spaces:
                options.append(action)

        return choice(options)

    def action_result(self, state: tuple, action: int) -> tuple:
        """
        Calculate the result of an action.

        Args:
            state (tuple): current state.
            action (int): proposed action.

        Returns:
            tuple: resulting space.
        """
        x, y = state[0]
        match action:
            case 0:  # Down
                new_space = (x, y + 1)
            case 1:  # Up
                new_space = (x, y - 1)
            case 2:  # Left
                new_space = (x - 1, y)
            case 3:  # Right
                new_space = (x + 1, y)
            case 4:
                new_space = state
        return new_space

    def best_action(self, state: tuple) -> int:
        """
        Best action for current state based on the q-table.

        Args:
            state (tuple): current state.

        Returns:
            int: best action from current state.
        """
        reward = 0
        action = -1
        for new_action in [0, 1, 2, 3]:
            new_reward = self.q[state, new_action]
            new_space = self.action_result(state, new_action)
            if new_reward > reward and new_space in self.spaces:
                reward = new_reward
                action = new_action

        return action

    def add_reward(self, observation, info):
        """
        Add a reward to the internal tables.

        Args:
            observation (np.ndarray): observation of the environment.
            info (dict): information dictionary.
        """
        new_pos = info["agent_pos"][self.agent_number]
        reward = 0
        # Agent is finished
        if info["agent_charging"][self.agent_number]:
            reward += 1000
        # Agent moved to a dirt space
        elif new_pos in self.dirt_left:
            reward += 10
            self.dirt_left.remove(new_pos)

        self.rewards.append(reward)

    def state_coverage(self) -> float:
        """
        Determines how many states have a valid action in their pi-table.
        """
        total_states = len(self.states)
        total_actions = sum(self.pi[state] != -1 for state in self.states)
        return total_actions / total_states

    def __str__(self):
        return f"MCAgent({self.agent_number}, {self.gamma})"
