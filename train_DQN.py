"""
Train DQN Agent.

Trains the DQN Agent and reports the results
"""
from pathlib import Path

from tqdm import trange
import numpy as np
import pandas as pd

try:
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

    from agents.DQNAgent import DQNAgent
    from world import Environment


def main(
    grid_paths: list[Path],
    dynamics_fp: list[Path | None],
    no_gui: bool,
    iters: int,
    fps: int,
    sigma: float,
    out_runs: Path,
    out_experiments: Path,
    random_seed: int,
):
    """
    Main loop of the program.

    Args:
        grid_paths (list[Path]): list of paths to the grids which we want to train on.
        dynamics_fp (list[Path  |  None]): list of dynamics files. None if the env should be static.
        no_gui (bool): should the gui be disabled.
        iters (int): max number of steps per episode.
        fps (int): target fps for the gui.
        sigma (float): randomness factor of the environment.
        out_runs (Path): output dir for the results.
        out_experiments (Path): output dir for the experiment results.
        random_seed (int): seed for the random locations during training.
    """    

    results = []
    for grid_file, dynamics_file in zip(grid_paths, dynamics_fp):
        room_name = grid_file.name

        # Set up the environment and reset it to its initial state
        env = Environment(grid_file, dynamics_file, no_gui, n_agents=1, agent_start_pos=None, sigma=0, reward_fn='custom',
                          target_fps=fps, random_seed=random_seed)
        _, info = env.get_observation()

        # Iterate through each agent for `iters` iterations
        TOTAL_ITERATIONS = 500

        # Set up the agents from scratch for every grid
        # Add your agents here
        agents = [
            DQNAgent(0, gamma=0.99, alpha=0.001, target_update_freq=1000,
                     grid_size=len(env.coord_to_array()[0]) * len(env.coord_to_array()[0][0]), epsilon=0.99),
            DQNAgent(0, gamma=0.99, alpha=0.001, target_update_freq=500,
                     grid_size=len(env.coord_to_array()[0]) * len(env.coord_to_array()[0][0]), epsilon=0.99),
            DQNAgent(0, gamma=0.99, alpha=0.0001, target_update_freq=1000,
                     grid_size=len(env.coord_to_array()[0]) * len(env.coord_to_array()[0][0]), epsilon=0.99),
            DQNAgent(0, gamma=0.99, alpha=0.0001, target_update_freq=500,
                     grid_size=len(env.coord_to_array()[0]) * len(env.coord_to_array()[0][0]), epsilon=0.99),
            DQNAgent(0, gamma=0.7, alpha=0.001, target_update_freq=1000,
                     grid_size=len(env.coord_to_array()[0]) * len(env.coord_to_array()[0][0]), epsilon=0.99),
            DQNAgent(0, gamma=0.7, alpha=0.001, target_update_freq=500,
                     grid_size=len(env.coord_to_array()[0]) * len(env.coord_to_array()[0][0]), epsilon=0.99),
            DQNAgent(0, gamma=0.7, alpha=0.0001, target_update_freq=1000,
                     grid_size=len(env.coord_to_array()[0]) * len(env.coord_to_array()[0][0]), epsilon=0.99),
            DQNAgent(0, gamma=0.7, alpha=0.0001, target_update_freq=500,
                     grid_size=len(env.coord_to_array()[0]) * len(env.coord_to_array()[0][0]), epsilon=0.99),
        ]


        for agent in agents:
            for i in range(TOTAL_ITERATIONS):
                print(i)
                for _ in trange(iters):
                    # Agent takes an action based on the latest observation and info
                    info['iteration'] = i/TOTAL_ITERATIONS
                    # print(env.coord_to_array()[0].flatten())
                    # print(info['dirt_vecs'])

                    input_nn = np.concatenate((np.array(env.coord_to_array()[0].flatten()), np.array(info['dirt_vecs'][0])))

                    action = agent.take_action(input_nn, info)

                    # The action is performed in the environment
                    _, reward, terminated, info = env.step([action])

                    input_nn = np.concatenate(
                        (np.array(env.coord_to_array()[0].flatten()), np.array(info['dirt_vecs'][0])))

                    agent.process_reward(input_nn, reward, action, terminated)

                    # If the agent is terminated, we reset the env.
                    if terminated:
                        break
                _, info, _ = env.reset()
                # print(world_stats)

            info['iteration'] = 0
            world_stats = Environment.evaluate_agent(grid_file, dynamics_file, [agent], 1000, out_runs, sigma, agent_start_pos=[(2, 2)])

            if dynamics_file is None:
                agent.save_model(dynamic=False)
            else:
                agent.save_model(dynamic=True)
            
            world_stats["start"] = (2, 2)
            world_stats["agent"] = str(agent)
            world_stats["room"] = room_name
            world_stats["sigma"] = sigma
            results.append(world_stats)
    
    results = pd.DataFrame.from_records(results)
    results.to_csv(out_experiments / "DQN_results.csv", index=False)


if __name__ == "__main__":
    main(
        grid_paths=[Path("grid_configs/warehouse_stat_5.grd"), Path("grid_configs/warehouse_dyn_5.grd")],
        dynamics_fp=[None, Path("dynamic_env_config/test.json")], 
        no_gui=True, 
        iters=1000, 
        fps=10, 
        sigma=0.3, 
        out_runs=Path("results/"), 
        out_experiments=Path("experiments/"), 
        random_seed=0
    )