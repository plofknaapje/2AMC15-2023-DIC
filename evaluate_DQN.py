"""Evaluate DQN.

Evaluates all DQN Agents and reports the results in the results folder.
"""
from pathlib import Path

try:
    from agents.DQNAgent import DQNAgent
    from world import EnvironmentDQN
except ModuleNotFoundError:
    import sys
    from os import pardir, path

    root_path = path.abspath(
        path.join(path.join(path.abspath(__file__), pardir), pardir)
    )

    if root_path not in sys.path:
        sys.path.extend(root_path)

    from agents.DQNAgent import DQNAgent
    from world import EnvironmentDQN


def main(
    grid_paths: list[Path],
    dynamics_fp: Path,
    fps: int,
    sigma: float,
    out_runs: Path,
    random_seed: int,
):
    """Main loop of the program."""

    for grid in grid_paths:
        # Set up the environment and reset it to its initial state
        env = EnvironmentDQN(
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
        ]

        for agent in agents:
            agent_name = f"DQN_models/model_updaterate{agent.target_update_freq}_gamma{agent.gamma}_alpha{agent.alpha}_dynamic.pt"
            agent.load_model(Path(agent_name))
            EnvironmentDQN.evaluate_agent(grid, dynamics_fp, [agent], 1000, out_runs, 0.1, agent_start_pos=[(2, 2)])


if __name__ == "__main__":
    main(
        grid_paths=[Path("grid_configs/warehouse_dyn_5.grd")],
        dynamics_fp=Path("dynamic_env_config/test.json"),
        fps=10,
        sigma=0,
        out_runs=Path("results/"),
        random_seed=0
    )
