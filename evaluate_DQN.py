from pathlib import Path
import pandas as pd

try:
    from agents.DQNAgent import DQNAgent
    from world.environment import EnvironmentDQN
except ModuleNotFoundError:
    import sys
    from os import pardir, path

    root_path = path.abspath(
        path.join(path.join(path.abspath(__file__), pardir), pardir)
    )

    if root_path not in sys.path:
        sys.path.extend(root_path)

    from agents.DQNAgent import DQNAgent
    from world.environment import EnvironmentDQN

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
    """Main loop of the program.

    Args:
        grid_paths (list[Path]): List of paths to the grid configurations.
        dynamics_fp (list[Path | None]): List of paths to the dynamics files. None if the grid should be static.
        no_gui (bool): Whether to run the simulation without GUI.
        iters (int): Number of iterations to run the evaluation.
        fps (int): Frames per second for the simulation.
        sigma (float): Standard deviation for the reward noise.
        out_runs (Path): Path to save the output evaluation results.
        out_experiments (Path): Path to save the summarised results.
        random_seed (int): Random seed for reproducibility of training.
    """
    results = []
    for grid_file, dynamics_file in zip(grid_paths, dynamics_fp):
        room_name = grid_file.name
        # Set up the environment and reset it to its initial state
        env = EnvironmentDQN(grid_file, dynamics_file, no_gui, n_agents=1, agent_start_pos=None, sigma=sigma,
                             reward_fn='custom', target_fps=fps, random_seed=random_seed)

        # Set up the agents from scratch for every grid
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

        if dynamics_file is None:
            dyn = "static"
        else:
            dyn = "dynamic"

        for agent in agents:
            # Construct the model weights path based on the agent's hyperparameters
            model_weights_path = Path(
                f"./DQN_models/model_updaterate{agent.target_update_freq}_gamma{agent.gamma}_alpha{agent.alpha}_{dyn}.pt"
            )
            try:
                agent.load_model(model_weights_path)
            except:
                print("No pretrained agent found!")
                continue

            # Evaluate the agent
            world_stats = EnvironmentDQN.evaluate_agent(grid_file, dynamics_file, [agent], iters, out_runs, sigma, agent_start_pos=[(2, 2)])
            print(f'Was tested on: /DQN_models/model_updaterate{agent.target_update_freq}_gamma{agent.gamma}_alpha{agent.alpha}_{dyn}.pt')

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
