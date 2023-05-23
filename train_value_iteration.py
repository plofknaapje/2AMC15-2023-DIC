"""Train.

Train your RL Agent in this file.
Feel free to modify this file as you need.

In this example training script, we use command line arguments. Feel free to
change this to however you want it to work.
"""
import os
from argparse import ArgumentParser
from pathlib import Path
from tqdm import trange

try:
    from agents.greedy_agent import GreedyAgent

    # Add your agents here
    from agents.null_agent import NullAgent
    from agents.random_agent import RandomAgent
    from agents.value_agent import ValueAgent
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

import pandas as pd

def train(
    grid_paths: list[Path],
    no_gui: bool,
    iters: int,
    fps: int,
    sigma: float,
    out_runs: Path,
    out_experiments: Path,
    random_seed: int,
):
    """
    Function which trains and evaluate the ValueIteration agents.

    Args:
        grid_paths (list[Path]): List of paths to the grids.
        no_gui (bool): Should the training be done in the GUI?
        iters (int): Number of iterations for training.
        fps (int): Target fps for the GUI.
        sigma (float): Sigma of the training environment.
        out_runs (Path): Output dir for the results.
        out_experiments (Path): Output dir for the experiment results.
        random_seed (int): Seed for the random locations during training.
    """

    results = []
    for grid_name in grid_paths:
        # Set up the environment and reset it to its initial state
        grid = Path(grid_name)
        room_name = grid_name.split("/")[1]

        env = Environment(
            grid,
            no_gui,
            n_agents=1,
            agent_start_pos=None,
            sigma=sigma,
            target_fps=fps,
            random_seed=random_seed,
        )
        obs, info = env.get_observation()
        # Add agents with different configurations here
        agents = [
            ValueAgent(0, gamma=0.9),
            ValueAgent(0, gamma=0.6)
        ]

        # Iterate through each agent for `iters` iterations
        for agent in agents:
            # Training loop
            for _ in trange(iters):
                # Agent takes an action based on the latest observation and info
                action = agent.take_action(obs, info)

                # The action is performed in the environment
                obs, reward, terminated, info = env.step([action])

                # If the agent is terminated, we reset the env.
                if terminated:
                    obs, info, world_stats = env.reset()
                    break

                agent.process_reward(obs, reward)
            obs, info, world_stats = env.reset()

            # Only go here AFTER training
            print("")
            print(str(agent))
            # Add starting spaces here
            if room_name == "test_1.grd":
                starts = [(8, 8), (1, 8), (8, 1), (4, 4)]

            if room_name == "test_2.grd":
                starts = [(1, 1), (1, 9), (10, 1), (7, 7)]

            for start in starts:
                # Add evaluation sigmas here
                for sigma in [0.0, 0.2]:
                    print(f"{agent}, start={start}, sigma={sigma}")
                    world_stats = Environment.evaluate_agent(grid, [agent], 1000, out_runs, sigma, agent_start_pos=[start])
                    world_stats["start"] = start
                    world_stats["agent"] = str(agent)
                    world_stats["room"] = room_name
                    world_stats["sigma"] = sigma
                    results.append(world_stats)

    results = pd.DataFrame.from_records(results)
    results.to_csv(out_experiments / "value_iteration_results.csv", index=False)


if __name__ == "__main__":
    train(
        grid_paths=["grid_configs/test_1.grd", "grid_configs/test_2.grd"],
        no_gui=True,
        iters=100,
        fps=10,
        sigma=0,
        out_runs=Path("results/"),
        out_experiments=Path("experiments/"),
        random_seed=0
    )
