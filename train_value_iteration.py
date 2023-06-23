"""
Trains the Value Iteration Agent and reports the result.
"""
from pathlib import Path
from tqdm import trange
import pandas as pd

try:
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

    from agents.value_agent import ValueAgent
    from world import Environment


def main(
    grid_paths: list[Path], 
    dynamics_paths: list[Path | None],
    no_gui: bool, 
    iters: int, 
    fps: int, 
    sigma: float, 
    out_runs: Path, 
    out_experiments: Path,
    random_seed: int
):
    """
    Function which trains and evaluate the ValueAgent agents.

    Args:
        grid_paths (list[Path]): list of paths to the grids.
        dynamics_paths (list[Path | None]): lost of paths to the dynamics files. None means static.
        no_gui (bool): should the training be done in the GUI?
        iters (int): number of iterations for training.
        fps (int): yarget fps for the GUI.
        sigma (float): sigma of the training environment.
        out_runs (Path): output dir for the results.
        out_experiments (Path): output dir for the experiment results.
        random_seed (int): seed for the random locations during training.
    """

    results = []
    for grid, dynamics in zip(grid_paths, dynamics_paths):
        # Set up the environment and reset it to its initial state
        room_name = grid.name

        env = Environment(grid, None, no_gui, n_agents=1, agent_start_pos=None, sigma=sigma, target_fps=fps, 
                          random_seed=random_seed, reward_fn='custom')
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
                    agent.reset()
                    break
                    
            obs, info, world_stats = env.reset()
            agent.reset()

            # Add starting spaces here
            start = (2, 2)
            world_stats = Environment.evaluate_agent(grid, None, [agent], 1000, out_runs, sigma=sigma, agent_start_pos=[start], 
                                                        random_seed=0, is_DQN=False)
            world_stats["start"] = start
            world_stats["agent"] = str(agent)
            world_stats["room"] = room_name
            world_stats["sigma"] = sigma
            results.append(world_stats)

    results = pd.DataFrame.from_records(results)
    results.to_csv(out_experiments / "value_iteration_results.csv", index=False)


if __name__ == "__main__":
    main(
        grid_paths=[Path("grid_configs/warehouse_stat_5.grd"), Path("grid_configs/warehouse_dyn_5.grd")], 
        dynamics_paths=[None, Path("dynamic_env_config/test.json")],
        no_gui=True, 
        iters=10, 
        fps=10, 
        sigma=0.3, 
        out_runs=Path("results/"), 
        out_experiments=Path("experiments/"), 
        random_seed=0
    )
