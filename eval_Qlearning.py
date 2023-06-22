from pathlib import Path
from agents.DQNAgent import DQNAgent
from world import Environment


def main(
    grid_paths: list[Path],
    dynamics_fp: list[Path],
    no_gui: bool,
    iters: int,
    fps: int,
    sigma: float,
    out_runs: Path,
    random_seed: int,
):
    """Main loop of the program.

    Args:
        grid_paths (list[Path]): List of paths to the grid configurations.
        dynamics_fp (list[Path]): List of paths to the dynamics files.
        no_gui (bool): Whether to run the simulation without GUI.
        iters (int): Number of iterations to run the evaluation.
        fps (int): Frames per second for the simulation.
        sigma (float): Standard deviation for the reward noise.
        out_runs (Path): Path to save the output evaluation results.
        random_seed (int): Random seed for reproducibility.
    """

    for dynamics_file, grid_file in zip(dynamics_fp, grid_paths):
        # Set up the environment and reset it to its initial state
        env = Environment(
            grid_file,
            dynamics_file,
            no_gui=no_gui,
            n_agents=1,
            agent_start_pos=None,
            sigma=sigma,
            reward_fn='custom',
            target_fps=fps,
            random_seed=random_seed,
        )
        obs, info = env.get_observation()

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

        dyn = {Path("dynamic_env_config/test.json"): 'static', Path("dynamic_env_config/test2.json"): 'dynamic'}

        for agent in agents:
            # Construct the model weights path based on the agent's hyperparameters
            model_weights_path = Path(
                f"./DQN_models/model_updaterate{agent.target_update_freq}_gamma{agent.gamma}_alpha{agent.alpha}_{dyn[dynamics_file]}.pt"
            )
            agent.load_model(model_weights_path)

            # Evaluate the agent
            Environment.evaluate_agent(grid_file, dynamics_file, [agent], iters, out_runs, 0.3, agent_start_pos=[(2, 2)])
            print(f'Was tested on: /DQN_models/model_updaterate{agent.target_update_freq}_gamma{agent.gamma}_alpha{agent.alpha}_{dyn[dynamics_file]}.pt')

            # Uncomment and modify the lines below as needed for additional evaluations
            # Environment.evaluate_agent(grid_file, dynamics_file, [agent], iters, out_runs, 0.2, agent_start_pos=[(1, 19)])
            # Environment.evaluate_agent(grid_file, dynamics_file, [agent], iters, out_runs, 0.0, agent_start_pos=[(2, 2)])
            # Environment.evaluate_agent(grid_file, dynamics_file, [agent], iters, out_runs, 0.0, agent_start_pos=[(1, 19)])


if __name__ == "__main__":
    main(
        grid_paths=[Path("grid_configs/warehouse_stat_5.grd"), Path("grid_configs/warehouse_dyn_5.grd")],
        dynamics_fp=[Path("dynamic_env_config/test2.json"), Path("dynamic_env_config/test.json")],
        no_gui=False,
        iters=1000,
        fps=10,
        sigma=0,
        out_runs=Path("results/"),
        random_seed=0
    )
