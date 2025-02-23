"""Value Iteration Agent.

Calculates the values of each state and determines the optimal policy based on
the most valuable state."""

import numpy as np
from itertools import chain, combinations
from time import time

from agents import BaseAgent

def powerset(iterable) -> list:
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)  # allows duplicate elements
    return [set for set in
            chain.from_iterable(combinations(s, r) for r in range(len(s)+1))]


class ValueAgent(BaseAgent):
    def __init__(self, agent_number: int, gamma: float, verbose=False, theta=0.1):
        """
        ValueAgent. Set agent parameters.

        Args:
            agent_number (int): the index of the agent in the environment.
            gamma (float): discount factor.
            verbose (bool): verbose output. Defaults to False.
            theta (float): minimal change per iteration. Defaults to 0.1.
        """
        super().__init__(agent_number)
        self.gamma = gamma
        self.theta = theta
        self.values = None
        self.verbose = verbose
        self.dirt_cleaned = []

    def process_reward(self, observation: np.ndarray, reward: float):
        """
        Placeholder function.

        Args:
            observation (np.ndarray): current environment grid.
            reward (float): reward.
        """
        pass

    def take_action(self, observation: np.ndarray, info: None | dict) -> int:
        """
        Default method which decides an action based on the environment.

        Args:
            observation (np.ndarray): current environment grid.
            info (None | dict): situation in current environment.

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

        # The robot keeps track of where it cleaned dirt.
        if agent_space in self.dirt_spaces and agent_space not in self.dirt_cleaned:
            self.dirt_cleaned.append(agent_space)
            self.dirt_cleaned.sort()
            if self.verbose:
                print(self.dirt_cleaned)

        state = (agent_space, tuple(self.dirt_cleaned))

        return self.generate_move(state)

    def generate_values(self, observation: np.ndarray):
        """
        Generate the values V(s) based on the provided observations matrix.
        Also generates some variables which store information about the structure
        of the observation grid.

        Args:
            observation (np.ndarray): Current environment grid.
        """
        cols, rows = observation.shape
        self.diagonal = cols + rows

        self.spaces = tuple(
            (i, j) for i in range(cols) for j in range(rows)
            if observation[i, j] in (0, 3, 4)
        )

        self.charge_spaces = tuple(
            space for space in self.spaces
            if observation[space] == 4
        )

        self.dirt_spaces = tuple(
            space for space in self.spaces
            if observation[space] == 3
        )

        dirt_configs = powerset(self.dirt_spaces)

        self.states = tuple(
            (space, dirt_cleaned)
            for space in self.spaces
            for dirt_cleaned in dirt_configs
        )

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
            
            # Optional progress update
            if self.verbose:
                print(f"Iter {i} with delta {delta:.2f}")
                i += 1

    def max_action(self, state: tuple) -> tuple[float, int]:
        """
        Try all possible actions from the current state and determine the action with
        the highest expected value.

        Args:
            state (tuple): current state of the agent.

        Returns:
            tuple[float, int]: value, action pair with the highest value.
        """
        value_action = []
    	
        # Build in reward function. Dirt is 10 points, charging is 100 points, hiting the charger is -1.
        for action in (0, 1, 2, 3, 4):
            new_state = self.action_outcome(state, action)
            new_space, dirt_cleaned = new_state

            if new_space == (-1, -1):
                continue
            elif new_space in self.dirt_spaces and new_space not in dirt_cleaned:
                value = 10
                dirt_cleaned = tuple(sorted(list(dirt_cleaned) + [new_space]))
            elif new_space in self.charge_spaces and dirt_cleaned == self.dirt_spaces:
                value = 100
            elif new_space in self.charge_spaces:
                value = -1
            else:
                value = 0
            new_value = value + self.values[new_state] * self.gamma
            value_action.append((new_value, action))

        return max(value_action)

    def action_outcome(self, state: tuple, action: int) -> tuple:
        """
        Determines the result of a certain action when taken in a certain place.

        Args:
            state (tuple): Current state of the agent.
            action (int): Action to be taken.

        Returns:
            tuple: Resulting state. ((-1, -1), ()) if the action is illegal.
        """
        # Col - row
        space, dirt_cleaned = state
        x, y = space

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
            return (new_space, dirt_cleaned)
        else:
            return ((-1, -1), ())

    def generate_move(self, state: tuple) -> int:
        """
        Generates the best move for a state.

        Args:
            state (tuple): Current state

        Returns:
            int: action to be taken.
        """
        return self.max_action(state)[1]
    
    def reset(self):
        """
        Reset dirt state memory
        """
        self.dirt_cleaned = []

    def __str__(self):
        return f"ValueAgent({self.agent_number}, {self.gamma})"
