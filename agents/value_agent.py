"""Value Iteration Agent.

Calculates the values of each state and determines the optimal policy based on
the most valuable state."""

import numpy as np

from agents import BaseAgent

from itertools import chain, combinations
from time import time


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)  # allows duplicate elements
    return [set for set in
            chain.from_iterable(combinations(s, r) for r in range(len(s)+1))]

class ValueAgent(BaseAgent):
    def __init__(self, agent_number: int, gamma: float, verbose=False, theta=0.1):
        """
        Set agent parameters.

        Args:
            agent_number (int): The index of the agent in the environment.
            gamma (float): Discount factor.
            verbose (bool): Verbose output. Defaults to False.
            theta (float): Minimal change per iteration. Defaults to 0.1.
        """
        super().__init__(agent_number)
        self.gamma = gamma
        self.theta = theta
        self.values = None
        self.verbose = verbose

    def process_reward(self, observation: np.ndarray, reward: float):
        """
        Placeholder function.

        Args:
            observation (np.ndarray): Current environment grid.
            reward (float): Reward.
        """
        pass

    def take_action(self, observation: np.ndarray, info: None | dict) -> int:
        """
        Default method which decides an action based on the environment.

        Args:
            observation (np.ndarray): Current environment grid.
            info (None | dict): Situation in current environment.

        Returns:
            int: action to be taken.
        """
        # If no values dict, generate it.
        if self.values is None:
            start = time()
            self.generate_values(observation)
            runtime = time() - start
            if self.verbose:
                print(f"Value Iteration took {runtime:.1f} seconds.")

        agent_space = info["agent_pos"][self.agent_number]

        dirt_left = tuple(space for space in self.spaces
                          if observation[space] == 3 and space != agent_space)

        state = (agent_space, dirt_left)

        return self.generate_move(state)

    def generate_values(self, observation: np.ndarray):
        """
        Generate the values V(s) based on the provided observations matrix.
        Also generates some variables which store information about the structure
        of the observation grid.

        Args:
            observation (np.ndarray): Current environment grid.

        Raises:
            ValueError:
        """
        cols, rows = observation.shape

        self.spaces = [
            (i, j) for i in range(cols) for j in range(rows)
            if observation[i, j] in (0, 3, 4)
        ]

        self.charge_spaces = [
            space for space in self.spaces
            if observation[space] == 4
        ]

        self.dirt_spaces = [
            space for space in self.spaces
            if observation[space] == 3
        ]

        dirt_configs = powerset(self.dirt_spaces)

        self.states = [
            (space, dirt_left)
            for space in self.spaces
            for dirt_left in dirt_configs
        ]

        self.values = {
            state: 0
            for state in self.states
        }

        if self.verbose and len(self.states) > 5000:
            print(f"{len(self.states)} states to optimize.")
            print("This number increases exponentially with the number of dirt spaces.")

        self.value_iteration()

    def value_iteration(self):
        """
        Value iteration implementation.
        Keeps optimizing self.values until change is smaller than theta.
        """
        delta = 2 * self.theta  # Initial delta to ensure loop starts
        i = 0
        while delta > self.theta:
            delta = 0
            for state in self.states:
                old_value = self.values[state]
                self.values[state] = self.max_action(state)[0]
                delta = max(delta, abs(self.values[state] - old_value))
            if self.verbose:
                print(f"Iter {i} with delta {delta:.2f}")
                i += 1

    def max_action(self, state: tuple[tuple, tuple]) -> tuple[float, int]:
        """
        Try all actions from the current state and determine the action with
        the highest expected value.

        Args:
            state (tuple[tuple, tuple]): current state of the agent.

        Returns:
            tuple[float, int]: value, action pair with the highest value.
        """
        value_action = []

        if state[0] in self.charge_spaces and len(state[1]) == 0:
            value_action.append((100.0, 4))

        for action in (0, 1, 2, 3, 4):
            new_state = self.action_outcome(state, action)
            new_space, dirt_left = new_state

            if new_space == (-1, -1):
                continue
            elif new_space in dirt_left:
                value = 10
            elif new_space in self.charge_spaces and len(dirt_left) == 0:
                value = 100
            elif new_space in self.charge_spaces:
                value = -1
            else:
                value = 0
            new_value = value + self.values[new_state] * self.gamma

            value_action.append((new_value, action))

        return max(value_action)

    def action_outcome(self, state: tuple[tuple, tuple], action: int) -> tuple[tuple, tuple]:
        """
        Determines the result of a certain action when taken in a certain place.

        Args:
            state (tuple[tuple, tuple]): Current state of the agent.
            action (int): Action to be taken.

        Returns:
            tuple[tuple, tuple]: Resulting state. (-1, -1) if the action is illegal.
        """
        # Col - row
        space, dirt_left = state
        x, y = space

        if space in dirt_left:
            temp = list(dirt_left)
            temp.remove(space)
            dirt_left = tuple(temp)

        match action:
            case 0:  # Down
                new_space = (x, y + 1)
            case 1:  # Up
                new_space = (x, y - 1)
            case 2:  # Left
                new_space = (x - 1, y)
            case 3:  # Right
                new_space = (x + 1, y)
            case 4:  # Stand still
                new_space = space

        if new_space in self.spaces:
            return (new_space, dirt_left)
        else:
            return ((-1, -1), ())

    def generate_move(self, state: tuple[tuple, tuple]) -> int:
        """
        Generates the best move for a state.

        Args:
            state (tuple[tuple, tuple]): Current state

        Returns:
            int: Action to be taken.
        """
        return self.max_action(state)[1]

    def __str__(self):
        return f"ValueAgent({self.agent_number}, {self.gamma})"
