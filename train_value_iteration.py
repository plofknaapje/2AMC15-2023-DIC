"""
Trains the Value Iteration Agent and reports the result.
"""
from pathlib import Path
from tqdm import trange

try:
    from agents.value_agent import ValueAgent

    from world.environment_ref import Environment

except ModuleNotFoundError:
    import sys
    from os import pardir, path

    root_path = path.abspath(
        path.join(path.join(path.abspath(__file__), pardir), pardir)
    )

    if root_path not in sys.path:
        sys.path.extend(root_path)

    from agents.value_agent import ValueAgent
    from world.environment_ref import Environment

import pandas as pd


def main(
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
    for grid in grid_paths:
        # Set up the environment and reset it to its initial state
        room_name = grid.name

        env = Environment(
            grid,
            no_gui=no_gui,
            n_agents=1,
            agent_start_pos=None,
            sigma=sigma,
            target_fps=fps,
            random_seed=random_seed,
            reward_fn='custom',
        )
        obs, info = env.get_observation()
        # Add agents with different configurations here
        agents = [
            ValueAgent(0, gamma=0.99),
            ValueAgent(0, gamma=0.7)
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
                    
            obs, info, world_stats = env.reset()

            # Add starting spaces here
            starts = [(2, 2)]

            for start in starts:
                world_stats = Environment.evaluate_agent(
                    grid, [agent], 1000, out_runs, sigma=0.0,
                    agent_start_pos=[start], random_seed=0)
                world_stats["start"] = start
                world_stats["agent"] = str(agent)
                world_stats["room"] = room_name
                world_stats["sigma"] = sigma
                results.append(world_stats)

    results = pd.DataFrame.from_records(results)
    print(results)
    results.to_csv(out_experiments / "value_iteration_results.csv", index=False)


if __name__ == "__main__":
    main(
        grid_paths=[Path("grid_configs\warehouse_stat_3.grd"),
                   Path("grid_configs\warehouse_stat_5.grd"),
                   Path("grid_configs\warehouse_stat_8.grd")],
        no_gui=True,
        iters=10,
        fps=10,
        sigma=0,
        out_runs=Path("results/"),
        out_experiments=Path("experiments/"),
        random_seed=0
    )
