"""Environment.

We define the grid environment for DIC in this file.
"""
import datetime
import random
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from time import sleep, time
from warnings import warn
from typing import Union
import numpy as np
from tqdm import trange
import json

# Custom modules may not be importable, depending on how you have set up your
# conda/pip/venv environment. Here we try to fix that by forcing the world to
# be in your python path. If it still doesn't work, come to a tutorial, look up
# how to fix module import errors, or ask ChatGPT.
try:
    from agents import BaseAgent
    from world.grid import Grid
    from world.gui import EnvironmentGUI
    from world.path_visualizer import visualize_path
except ModuleNotFoundError:
    import sys
    from os import pardir, path


    root_path = path.abspath(
        path.join(path.join(path.abspath(__file__), pardir), pardir)
    )


    if root_path not in sys.path:
        sys.path.append(root_path)


    from agents import BaseAgent
    from world.grid import Grid
    from world.gui import EnvironmentGUI
    from world.path_visualizer import visualize_path

class Environment:
    def __init__(
        self,
        grid_fp: Path,
        dynamics_fp: Path | None,
        no_gui: bool = False,
        n_agents: int = 1,
        sigma: float = 0.0,
        agent_start_pos: list[tuple[int, int]] = None,
        reward_fn: callable = None,
        target_fps: int = 30,
        random_seed: int | float | str | bytes | bytearray | None = 0
    ):
        """Creates the grid environment for the robot vacuum.

        Creates a Grid environment from the provided grid file. The number of
        agents is variable, allowing for multi-agent environments. If any
        start positions are provided, then the number of positions must be
        equal to the number of agents.

        This environment follows the general principles of reinforcment
        learning. It can be though of as a function E : action -> observation
        where E is the environment represented as a function.

        Args:
            grid_fp: Path to the grid file to use.
            no_gui: True if no GUI is desired.
            n_agents: The number of agents this environment should support.
            sigma: The stochasticity of the environment. The probability that
                an agent makes the move that it has provided as an action is
                calculated as 1-sigma.
            agent_start_pos: List of tuples of where each agent should start.
                If None is provided, then a random start position for each
                agent is used.
            reward_fn: Custom reward function to use. It should have a
                signature of func(grid: Grid, info: dict) -> float. See the
                default reward function in this class for an example.
            target_fps: How fast the simulation should run if it is being shown
                in a GUI. This is a target, not the actual speed. If in
                no_gui mode, then the simulation will run as fast as
                possible. We may set a low FPS so we can actually see what's
                happening. Set to 0 or less to unlock FPS.
            random_seed: The random seed to use for this environment. If None
                is provided, then the seed will be set to 0.
        """
        random.seed(random_seed)
        self.sigma = sigma
        if not grid_fp.exists():
            raise FileNotFoundError(f"Grid {grid_fp} does not exist.")

        # Load the grid from the file
        self.grid_fp = grid_fp
        
        # Set up the environment as a blank state.
        self.grid = None
        self.no_gui = no_gui
        if target_fps <= 0:
            self.target_spf = 0.0
        else:
            self.target_spf = 1.0 / target_fps
        self.gui = None
        
        # Set up initial agent positions
        self.n_agents = n_agents  # Number of active agents
        self.agent_pos = None  # Current agent positions
        self.agent_start_pos = agent_start_pos  # Where agents initially start
        self.agent_done = [False] * n_agents
        
        # Set up reward function
        if reward_fn is None:
            warn("No reward function provided. Using default reward.")
            self.reward_fn = self._default_reward_function
        else:
            self.reward_fn = self._custom_reward_function
        self.info = self._reset_info()
        self.world_stats = self._reset_world_stats()
        
        self.environment_ready = False

        # Set up for dynamics
        self.dynamics_fp = dynamics_fp
        if self.dynamics_fp is not None:
            with open(self.dynamics_fp) as file:
                self.dynamic_obs = json.load(file)
            self.truck_pos = []
            self.truck_track = []

            self.persons_pos = []

        self.reset()

        dirt_tiles = [
            (i, j) for i in range(self.grid.n_cols) for j in range(self.grid.n_rows)
            if self.grid.cells[i, j] == 3
        ]
        if len(dirt_tiles) >= 30:
            raise ValueError("Environment does not support more than 30 dirt!")

        self.dirt_dict = {coord: i for i, coord in enumerate(dirt_tiles)}

    def check_dynamics(self):
        """Check feasibility of the obstacles in the grid if there are any moveable objects.

        Returns an error if something is wrong

        First checks whether the objects stay in bound of the grid.

        Then it checks if the space in moves in is free from other obstacles/dirt/chargers

        Every truck has an id, trajectory and a width and height
        """
        # Check trucks dynamics
        self.truck_pos = []
        trucks = self.dynamic_obs['trucks']

        grid = deepcopy(self.grid.cells)

        for truck in trucks:
            width = truck['width']
            height = truck['height']
            trajectory = truck['trajectory']

            # Checks if trajectory is free from dirt/obstacles or chargers
            for pos in trajectory:
                for i in range(width):
                    for j in range(height):
                        if grid[pos[0]+i, pos[1]+j] != 0:
                            raise ValueError('One or more obstacles collide or interfere with dirt/charger')

            # Initialize truck position and place it on the grid
            self.truck_pos.append(0)
            for i in range(2):
                for j in range(2):
                    self.grid.cells[trajectory[0][0]+i, trajectory[0][1] + j] = 2

        # Check person dynamics
        self.persons_pos = []
        persons = self.dynamic_obs['persons']
        for person in persons:
            start_pos = person['position']
            if self.grid.cells[start_pos[0], start_pos[1]] != 0:
                raise ValueError('Person interferes with obstacle/dirt/charger')
            else:
                self.grid.cells[start_pos[0], start_pos[1]] = 2
                self.persons_pos.append(start_pos)

    def dynamics(self):
        """Move the trucks in the grid if.

        Tries to move the truck to the next position in the trajectory
        If robot is in the way the truck stays in the same position

        Every truck has an id, trajectory and a width and height
        """
        # update person position
        for p, person in enumerate(self.persons_pos):
            arr = [0, 1, 2, 3]
            n = len(arr)
            for i in range(n - 1, 0, -1):
                j = random.randint(0, i)
                arr[i], arr[j] = arr[j], arr[i]

            change = True
            for i in arr:
                if change:
                    change = self.move_person(i, p, person)


        # update forklift truck position
        trucks = self.dynamic_obs['trucks']
        agent_pos = self.info["agent_pos"]

        for truck_number, truck in enumerate(trucks):
            width = truck['width']
            height = truck['height']
            trajectory = truck['trajectory']

            # Determine the next position in trajectory of the truck
            position = self.truck_pos[truck_number]
            next_pos = (position+1) % len(trajectory)

            # Check if robot is in next position in the trajectory
            possible = True
            for i in range(width):
                for j in range(height):
                    if agent_pos[0] == (trajectory[next_pos][0] + i, trajectory[next_pos][1] + j):
                        possible = False
                    for person in self.persons_pos:
                        if [trajectory[next_pos][0] + i, trajectory[next_pos][1] + j] == person:
                            print('person block')
                            possible = False

            # If robot is not in the way --> update truck to new position
            if possible:
                # Set the new position of the truck
                self.truck_pos[truck_number] = next_pos

                # Remove truck from old position
                for i in range(width):
                    for j in range(height):
                        self.grid.cells[trajectory[position][0]+i, trajectory[position][1]+j] = 0

                # Add truck in new position
                for i in range(width):
                    for j in range(height):
                        self.grid.cells[trajectory[next_pos][0]+i, trajectory[next_pos][1]+j] = 2


    def move_person(self, i, p, person):
        if i == 0 and self.grid.cells[person[0], person[1] + 1] == 0:
            self.grid.cells[person[0], person[1] + 1] = 2
            self.grid.cells[person[0], person[1]] = 0
            self.persons_pos[p] = [person[0], person[1] + 1]
            return False

        if i == 1 and self.grid.cells[person[0], person[1] - 1] == 0:
            self.grid.cells[person[0], person[1] - 1] = 2
            self.grid.cells[person[0], person[1]] = 0
            self.persons_pos[p] = [person[0], person[1] - 1]
            return False

        if i == 2 and self.grid.cells[person[0]+1, person[1]] == 0:
            self.grid.cells[person[0]+1, person[1]] = 2
            self.grid.cells[person[0], person[1]] = 0
            self.persons_pos[p] = [person[0]+1, person[1]]
            return False
        if i == 3 and self.grid.cells[person[0]-1, person[1]] == 0:
            self.grid.cells[person[0]-1, person[1]] = 2
            self.grid.cells[person[0], person[1]] = 0
            self.persons_pos[p] = [person[0]-1, person[1]]
            return False
        return True



    def _reset_info(self) -> dict:
        """Resets the info dictionary.

        info is a list of stats of the most recent step. It contains how many
        dirt tiles were cleaned, a list of if the agent moved, and a list of
        if an agent is done, i.e., succesfully moved onto the charger.
        agent_moved and agent_done are a boolean list.

        For example, if agent 0 moved, agent 1 failed to move, and agent 2
        moved, then agent_moved = [True, False, True].

        Same thing for agent_charging. agent_charging is True if the agent
        moved to the charger this turn

        Similarly, the index of dirt_cleaned is the number of dirt tiles
        cleaned by the agent at that index. If agent 0 cleaned 1 dirt tile,
        agent 1 cleaned 0 dirt tiles, and agent 2 cleaned 0 dirt tiles, then
        dirt_cleaned would be [1, 0, 0]
        """
        return {
            "dirt_cleaned": [0] * self.n_agents,
            "agent_moved": [False] * self.n_agents,
            "agent_charging": self.agent_done,
            "agent_pos": self.agent_pos,
            "agent_pos_arr": [None] * self.n_agents,
            "dirt_vecs": [np.zeros(30)] * self.n_agents
        }

    @staticmethod
    def _reset_world_stats() -> dict:
        return {
            "total_dirt_cleaned": 0,
            "total_steps": 0,
            "total_agent_moves": 0,
            "total_agents_at_charger": 0,
            "total_failed_moves": 0,
            "steps_per_dirt": 0,
            "failed_moves_fraction": 0,
            "total_reward": 0
        }

    def _initialize_agent_pos(self):
        """Initializes agent position from the givin initial variables.

        If None is given as the agent_start_pos, choose a position at random to
        place to the agent. If it is filled, try again until it finds a place
        that has not been filled yet. This assumes a mostly empty grid.
        """
        agent_pos = []
        if self.agent_start_pos is not None:
            # We try placing each agent at every requested position.
            for i in range(self.n_agents):
                pos = (self.agent_start_pos[i][0], self.agent_start_pos[i][1])
                if self.grid.cells[pos] == 0:
                    # Cell is empty. We can place the agent there.
                    agent_pos.append(pos)
                else:
                    # Agent is placed on walls/obstacle/dirt/charger
                    raise ValueError(
                        "Attempted to place agent on top of wall or " "charger"
                    )
            self.agent_pos = deepcopy(self.agent_start_pos)
        else:
            # No positions were given. We place agents randomly.
            warn(
                "No initial agent positions given. Randomly placing agents "
                "on the grid."
            )
            for _ in range(self.n_agents):
                # First get all empty positions
                zeros = np.where(self.grid.cells == 0)
                idx = random.randint(0, len(zeros[0]) - 1)
                agent_pos.append((zeros[0][idx], zeros[1][idx]))
            self.agent_pos = agent_pos

    def get_observation(self):
        """Gets the current observation and information.

        Returns:
            - observation as an np.ndarray
            - info as a dict with keys ['dirt_cleaned', 'agent_moved',
              'agent_charging', 'agent_pos', 'agent_pos_arr', 'dirt_vecs']
        """
        # Only update once an agent wants the observations.
        self.info["agent_pos_arr"] = self.coord_to_array()
        return self.grid.cells, self.info

    def coord_to_array(self) -> [np.ndarray]:
        # Transform coordinate representation to a 1 on a 0-matrix.
        arrs = []
        for pos in self.info["agent_pos"]:
            matrix = np.zeros_like(self.grid.cells)
            matrix[pos] = 1
            arrs.append(matrix)
        return arrs

    def reset(self, **kwargs) -> [np.ndarray, dict, dict]:
        """Reset the environment to an initial state.

        This is to reset the environment. You can fit it keyword arguments
        which will overwrite the initial arguments provided when initializing
        the environment.

        Example:
        >>> fp = Path("../grid_configs/base-room-1.grid")
        >>> e = Environment(fp, False, 1, 0., None)
        >>> # Get the initial observation
        >>> observation, env_info = e.get_observation()
        >>> # Reset the environment, but for this training episode, we want
        >>> # to use 2 agents.
        >>> observation, env_info, world_stats = e.reset(n_agents=2)

        Args:
            **kwargs: possible keyword options are the same as those for
                the environment initializer.
        Returns:
            - observation as an np.ndarray
            - info as a dict with keys ['dirt_cleaned', 'agent_moved',
                'agent_charging', 'agent_pos']
            - last run stats as a dict with keys ['total_dirt_cleaned',
                'total_steps', 'total_agent_moves', 'total_agents_at_charger',
                'total_failed_moves'].
        """
        world_stats = deepcopy(self.world_stats)
        for k, v in kwargs.items():
            # Go through each possible keyword argument.
            match k:
                case "grid_fp":
                    self.grid_fp = v
                case "n_agents":
                    self.n_agents = v
                    self.agent_pos = None
                    self.agent_start_pos = None
                case "agent_start_pos":
                    self.agent_start_pos = v
                case "no_gui":
                    self.no_gui = v
                case "target_fps":
                    self.target_spf = 1.0 / v
                case _:
                    raise ValueError(
                        f"{k} is not one of the possible " f"keyword arguments."
                    )
                    
        if self.agent_start_pos is not None:
            if len(self.agent_start_pos) != self.n_agents:
                raise ValueError(
                    f"Number of agents {self.n_agents} does not "
                    f"agree with number of starting positions "
                    f"{len(self.agent_start_pos)}."
                )
                
        self.grid = Grid.load_grid_file(self.grid_fp)
        if self.dynamics_fp is not None:
            self.check_dynamics()

        self._initialize_agent_pos()
        self.info = self._reset_info()
        self.world_stats = self._reset_world_stats()
        if not self.no_gui:
            self.gui = EnvironmentGUI(self.grid.cells.shape)
            self.gui.reset()
        else:
            if self.gui is not None:
                self.gui.close()
                
        self.environment_ready = True
        self.agent_done = [False] * len(self.agent_done)
        return self.grid.cells, self.info, world_stats
    
    def _move_agent(self, new_pos: tuple[int, int], agent_id: int):
        """Moves the agent, if possible.

        If possible, the agents' position is changed in the agent_pos array.
        If not, it is left untouched.

        Args:
            new_pos: The new position of the agent.
            agent_id: The id of the agent. This is its index in the list of
                agent positions.
        """
        # First check if any other agent is on that tile position
        for other_agent_id in range(len(self.agent_pos)):
            # Don't check against self
            if agent_id == other_agent_id:
                continue
            if new_pos == self.agent_pos[other_agent_id]:
                # print("Move goes into another agent")
                self.world_stats["total_failed_moves"] += 1
                return

        match self.grid.cells[new_pos]:
            case 0:  # Moved to an empty tile
                self.agent_pos[agent_id] = new_pos
                self.info["agent_moved"][agent_id] = True
                self.world_stats["total_agent_moves"] += 1
            case 1 | 2:  # Moved to a wall or obstacle
                # print(
                #     f"Agent {agent_id} tried to move into a wall at {new_pos} from {self.agent_pos[agent_id]}"
                # )
                self.world_stats["total_failed_moves"] += 1
                pass
            case 3:  # Moved to a dirt tile
                self.agent_pos[agent_id] = new_pos
                self.grid.cells[new_pos] = 0
                self.info["dirt_cleaned"][agent_id] += 1
                self.world_stats["total_dirt_cleaned"] += 1
                self.info["agent_moved"][agent_id] = True
                self.world_stats["total_agent_moves"] += 1
                # Record the encoded value of all cleaned dirts
                self.info["dirt_vecs"][agent_id][self.dirt_dict[new_pos]] = 1
            case 4:  # Moved to the charger
                # Moving to charger is only permitted if the room is clean.
                # NOTE: This is a pending design decision.
                if self.grid.sum_dirt() == 0:
                    self.agent_pos[agent_id] = new_pos
                    self.agent_done[agent_id] = True
                    self.info["agent_charging"][agent_id] = True
                    self.world_stats["total_agents_at_charger"] += 1
                # Otherwise, the agent can't move and nothing happens
                else:
                    # print("Room is not clean yet so charging is not allowed.")
                    self.world_stats["total_failed_moves"] += 1
            case _:
                raise ValueError(
                    f"Grid is badly formed. It has a value of "
                    f"{self.grid.cells[new_pos]} at position "
                    f"{new_pos}."
                )
                
    def step(self, actions: list[int]) -> [np.ndarray, float, bool, dict]:
        """This function makes the agent take a step on the grid.

        Actions are provided as a list of integers. The integer values are:
            - 0: Move down
            - 1: Move up
            - 2: Move left
            - 3: Move right
            - 4: Stand still

        Args:
            actions: List of integers representing the action each agent should
                take. The index of the action corresponds to which agent did
                which action.

        Returns:
            0) Current grid cells,
            1) The reward for the agent,
            2) If the terminal state has been reached, and
            3) State information.
        """
        self.world_stats["total_steps"] += 1
        is_single_step = False
        if not self.no_gui:
            start_time = time()
            while self.gui.paused:
                # If the GUI is paused but asking to step, then we step
                if self.gui.step:
                    is_single_step = True
                    self.gui.step = False
                    break
                # Otherwise, we render the current state only
                paused_info = self._reset_info()
                paused_info["agent_moved"] = [True] * self.n_agents
                self.gui.render(
                    self.grid.cells, self.agent_pos, paused_info, is_single_step
                )
                
        if not self.environment_ready:
            raise ValueError(
                "reset() has not been called yet. "
                "The environment still needs to be initialized."
            )
        # Verify that the number of actions and the number of agents is the
        # same
        if len(actions) != self.n_agents:
            raise ValueError(
                f"Number of actions provided is {len(actions)}, "
                f"but the number of agents is {self.n_agents}."
            )
            
        dirt_vecs = self.info["dirt_vecs"]
        self.info = self._reset_info()
        self.info["dirt_vecs"] = dirt_vecs
        
        max_x = self.grid.n_cols - 1
        max_y = self.grid.n_rows - 1
        
        for i, action in enumerate(actions):
            if self.agent_done[i]:
                # The agent is already on the charger, so it is done.
                continue
            
            # Add stochasticity into the agent action
            val = random.random()
            if val > self.sigma:
                actual_action = action
            else:
                actual_action = random.randint(0, 4)
            match actual_action:
                case 0:  # Move down
                    new_pos = (
                        self.agent_pos[i][0],
                        min(max_y, self.agent_pos[i][1] + 1),
                    )
                case 1:  # Move up
                    new_pos = (self.agent_pos[i][0], max(0, self.agent_pos[i][1] - 1))
                case 2:  # Move left
                    new_pos = (max(0, self.agent_pos[i][0] - 1), self.agent_pos[i][1])
                case 3:  # Move right
                    new_pos = (
                        min(max_x, self.agent_pos[i][0] + 1),
                        self.agent_pos[i][1],
                    )
                case 4:  # Stand still
                    new_pos = (self.agent_pos[i][0], self.agent_pos[i][1])
                case _:
                    raise ValueError(
                        f"Provided action {action} for agent {i} "
                        f"is not one of the possible actions."
                    )
            self._move_agent(new_pos, i)

        # Update the grid with the new agent positions and calculate the reward
        reward = self.reward_fn(self.grid, self.info)

        self.info["agent_pos_arr"] = self.coord_to_array()
        
        # Get total reward
        self.world_stats["total_reward"] += reward
        
        terminal_state = sum(self.agent_done) == self.n_agents
        if terminal_state:
            self.environment_ready = False

        if terminal_state:
            terminal_state = 1
        else:
            terminal_state = 0
            
        if not self.no_gui:
            time_to_wait = self.target_spf - (time() - start_time)
            if time_to_wait > 0:
                sleep(time_to_wait)
            self.gui.render(self.grid.cells, self.agent_pos, self.info, is_single_step)

        # After agent makes a move change the obstacles from positions
        if self.dynamics_fp is not None:
            self.dynamics()
            
        return self.grid.cells, reward, terminal_state, self.info

    @staticmethod
    def _default_reward_function(grid: Grid, info: dict) -> float:
        """This is the default reward function.

        This is a very simple default reward function. It simply checks if any
        dirt tiles were cleaned during the step and provides a reward equal to
        the total number of dirt tiles cleaned.

        Any custom reward function must also follow the same signature, meaning
        it must be written like `reward_name(grid, info)`.

        Args:
            grid: The grid the agent is moving on, in case that is needed by
                the reward function.
            info: The world info, in case that is needed by the reward function.

        Returns:
            A single floating point value representing the reward for a given
            action.
        """
        return float(sum(info["dirt_cleaned"]))
    
    @staticmethod
    def _custom_reward_function(grid: Grid, info: dict) -> float:
        """This is the custom reward function.

        Args:
            grid: The grid the agent is moving on, in case that is needed by
                the reward function.
            info: The world info, in case that is needed by the reward function.

        Returns:
            A single floating point value representing the reward for a given
            action.
        """
        dirt_reward = sum(info["dirt_cleaned"]) * 50
        
        if info["agent_moved"] == [False] and info["agent_charging"][0] != True:
            bumped_reward = -10
        else:
            bumped_reward = 0
            
        if info["agent_moved"] == [True] and dirt_reward == 0:
            moving_reward = -1
        else:
            moving_reward = 0
            
        if grid.sum_dirt() == 0 and info["agent_charging"][0]:
            charging_reward = 100
        elif info["agent_charging"][0]:
            charging_reward = -1
        else:
            charging_reward = 0
            
        # print(grid.sum_dirt(), info["agent_charging"][0])
        # print('DIRT REWARD:', dirt_reward)
        # print('BUMPED REWARD:', bumped_reward)
        # print('CHARGING REWARD:', charging_reward)
        # print('MOVED REWARD: ', moving_reward)
            
        total_reward = dirt_reward + bumped_reward + charging_reward + moving_reward
        
        return total_reward 
    
    @staticmethod
    def evaluate_agent(
        grid_fp: Path,
        dynamics_fp: Path | None,
        agents: list[BaseAgent],
        max_steps: int,
        out_runs: Path,
        sigma: float = 0.0,     #test with 2 values
        agent_start_pos: list[tuple[int, int]] = None,
        random_seed: int | float | str | bytes | bytearray = 0,
        show_images: bool = False,
        is_DQN: bool = True
    ):
        """Evaluates a single trained agent's performance.

        What this does is it creates a completely new environment from the
        provided grid and does a number of steps _without_ processing rewards
        for the agent. This means that the agent doesn't learn here and simply
        provides actions for any provided observation.

        For each evaluation run, this produces a statistics file in the out
        directory which is a txt. This txt contains the values:
        [ `total_dirt_cleaned`, `total_steps`, `total_retraced_steps`,
        `total_agents_at_charger`, `total_failed_moves`]

        For each agent, this produces an image file in the given out directory
        containing the path of the agent throughout the run.

        Args:
            grid_fp: Path to the grid file to use.
            dynamics_fp: Path to the dynamics file to use.
            agents: A list of trained agents to evaluate.
            max_steps: Max number of steps to take for each agent.
            out_runs: Where to save the results.
            sigma: The stochasticity of the environment. The probability that
                an agent makes the move that it has provided as an action is
                calculated as 1-sigma.
            agent_start_pos: List of tuples of where each agent should start.
                If None is provided, then a random start position for each
                agent is used.
            random_seed: The random seed to use for this environment. If None
                is provided, then the seed will be set to 0.
            show_images: Whether to show the images at the end of the
                evaluation. If False, only saves the images.
            is_DQN: Agent is DQN. Defaults to True.
        """
        if not out_runs.exists():
            warn(
                "Evaluation output directory does not exist. Creating the " "directory."
            )
            out_runs.mkdir(parents=True, exist_ok=True)
        env = Environment(
            grid_fp=grid_fp,
            dynamics_fp=dynamics_fp,
            no_gui=True,
            n_agents=len(agents),
            sigma=sigma,
            agent_start_pos=agent_start_pos,
            target_fps=-1,
            random_seed=random_seed,
            reward_fn='custom'
        )
        obs, info = env.get_observation()
        
        initial_grid = np.copy(obs)

        # Set initial positions for the agent
        agent_paths = [[pos] for pos in info["agent_pos"]]

        for _ in trange(
            max_steps, desc=f"Evaluating agent" f"{'s' if len(agents) > 1 else ''}"
        ):
            if is_DQN:
                # Get the agent actions
                input_nn = np.concatenate(
                    (np.array(env.coord_to_array()[0].flatten()), np.array(info['dirt_vecs'][0])))

                actions = [agent.take_action_eval(input_nn, info) for agent in agents]
            else:
                actions = [agent.take_action(obs, info) for agent in agents]
                
            # Take a step in the environment
            obs, _, terminated, info = env.step(actions)

            # Save the new agent locations
            for i, pos in enumerate(info["agent_pos"]):
                agent_paths[i].append(pos)

            if terminated:
                break

        summed_dirt = env.grid.sum_dirt()
        obs, info, world_stats = env.reset()

        world_stats["dirt_remaining"] = summed_dirt

        # Get custom evaluation metrics
        if world_stats["total_dirt_cleaned"]:
            world_stats["steps_per_dirt"] = (world_stats["total_agent_moves"] + world_stats["total_failed_moves"]) \
                / world_stats["total_dirt_cleaned"]
        else:
            world_stats["steps_per_dirt"] = -1
        world_stats["failed_moves_fraction"] = world_stats["total_failed_moves"] \
            / (world_stats["total_agent_moves"] + world_stats["total_failed_moves"])

        # Generate path images
        path_images = visualize_path(initial_grid, agent_paths)

        print("Evaluation complete. Results:")
        # File name is the current date and time
        file_name = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
        out_fp = out_runs / f"{file_name}.txt"
        with open(out_fp, "w") as f:
            for key, value in world_stats.items():
                f.write(f"{key}: {value}\n")
                print(f"{key}: {value}")

        # Save the images
        for i, img in enumerate(path_images):
            img_name = f"{file_name}_agent-{i}"
            out_fp = out_runs / f"{img_name}.png"
            img.save(out_fp)
            if show_images:
                img.show(f"Agent {i} Path Frequency")

        return world_stats


if __name__ == "__main__":
    # This is sample code to test a single grid.
    base_grid_fp = Path("grid_configs/rooms-1.grd")
    envi = Environment(base_grid_fp, False, 1, target_fps=5)
    observe, inf = envi.get_observation()
    print(inf)

    # Load the random agent
    from agents.random_agent import RandomAgent

    test_agent = RandomAgent(agent_number=0)

    # Take 1000 steps with the GUI
    for t in trange(1000):
        act = [test_agent.take_action(observe, inf)]
        observe, r, term_state, inf = envi.step(act)
        if term_state:
            break

    # Take 10000 steps without the GUI
    observe, inf, stats = envi.reset(no_gui=True)
    # print(stats)
    for t in trange(100000):
        act = [test_agent.take_action(observe, inf)]
        observe, r, term_state, inf = envi.step(act)
        if term_state:
            break

    # print(envi.reset()[2])  # Print the world stats
