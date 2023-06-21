"""Train.

Trains the Q-Learning Agent and reports the results
"""
from argparse import ArgumentParser
from pathlib import Path

from tqdm import trange
import numpy as np

try:
    from agents.greedy_agent import GreedyAgent

    # Add your agents here
    from agents.null_agent import NullAgent
    from agents.random_agent import RandomAgent
    from agents.value_agent import ValueAgent
    from agents.QLearn_agent import QLearnAgent
    from agents.DQNAgent import DQNAgent
    from world import Environment
except ModuleNotFoundError:
    import sys
    from os import pardir, path

    root_path = path.abspath(
        path.join(path.join(path.abspath(__file__), pardir), pardir)
    )

    if root_path not in sys.path:
        sys.path.extend(root_path)

    from agents.greedy_agent import GreedyAgent

    # Add your agents here
    from agents.null_agent import NullAgent
    from agents.random_agent import RandomAgent
    from agents.value_agent import ValueAgent
    from world import Environment


def main(
    grid_paths: list[Path],
    dynamics_fp: Path,
    no_gui: bool,
    iters: int,
    fps: int,
    sigma: float,
    out_runs: Path,
    out_experiments: Path,
    random_seed: int,
):
    """Main loop of the program."""

    for grid in grid_paths:
        # Set up the environment and reset it to its initial state
        env = Environment(
            grid,
            dynamics_fp=dynamics_fp,
            no_gui=True,
            n_agents=1,
            agent_start_pos=None,
            sigma=sigma,
            reward_fn='custom',
            target_fps=fps,
            random_seed=random_seed,
        )
        obs, info = env.get_observation()



        # Set up the agents from scratch for every grid
        # Add your agents here
        agents = [
            DQNAgent(0, 0.99, len(env.coord_to_array()[0])*len(env.coord_to_array()[0][0]), epsilon=0.99)           
        ]


        for agent in agents:
            agent.load_model(Path("DQN_models/model_updaterate1000_gamma0.99_alpha0.001.pt"))
            Environment.evaluate_agent(grid, dynamics_fp, [agent], 1000, out_runs, 0.1, agent_start_pos=[(2, 2)])
            # Environment.evaluate_agent(grid, [agent], 1000, out_runs, 0.0, agent_start_pos=[(1, 8)])
            # Environment.evaluate_agent(grid, [agent], 1000, out_runs, 0.0, agent_start_pos=[(8, 1)])
            # Environment.evaluate_agent(grid, [agent], 1000, out_runs, 0.0, agent_start_pos=[(8, 8)])
            #
            # Environment.evaluate_agent(grid, [agent], 1000, out_runs, 0.25, agent_start_pos=[(1, 1)])
            # Environment.evaluate_agent(grid, [agent], 1000, out_runs, 0.25, agent_start_pos=[(1, 8)])
            # Environment.evaluate_agent(grid, [agent], 1000, out_runs, 0.25, agent_start_pos=[(8, 1)])
            # Environment.evaluate_agent(grid, [agent], 1000, out_runs, 0.25, agent_start_pos=[(8, 8)])
            

if __name__ == "__main__":
    main(
        grid_paths=[Path("grid_configs/warehouse_dyn_3.grd")],
        dynamics_fp=Path("dynamic_env_config/test.json"),
        no_gui=False,
        iters=1000,
        fps=10,
        sigma=0,
        out_runs=Path("results/"),
        out_experiments=Path("experiments/"),
        random_seed=0
    )
