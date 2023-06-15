"""Train.

Trains the Q-Learning Agent and reports the results
"""
from argparse import ArgumentParser
from pathlib import Path

from tqdm import trange

try:
    from agents.greedy_agent import GreedyAgent

    # Add your agents here
    from agents.null_agent import NullAgent
    from agents.random_agent import RandomAgent
    from agents.value_agent import ValueAgent
    from agents.QLearn_agent import QLearnAgent
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
            no_gui=False,
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
            QLearnAgent(0, gamma=0.9, epsilon=0.4),
            QLearnAgent(0, gamma=0.6, epsilon=0.4)
        ]

        # Iterate through each agent for `iters` iterations
        TOTAL_ITERATIONS = 20000

        for agent in agents:
            for i in range(TOTAL_ITERATIONS):
                print(i)
                for _ in trange(iters):
                    # Agent takes an action based on the latest observation and info
                    info['iteration'] = i/TOTAL_ITERATIONS
                    action = agent.take_action(obs, info)

                    # The action is performed in the environment
                    obs, reward, terminated, info = env.step([action])

                    agent.process_reward(action, reward)

                    # If the agent is terminated, we reset the env.
                    if terminated:
                        break
                obs, info, world_stats = env.reset()
                print(world_stats)

            info['iteration'] = 0
            Environment.evaluate_agent(grid, [agent], 1000, out_runs, 0.0, agent_start_pos=[(1, 1)])
            Environment.evaluate_agent(grid, [agent], 1000, out_runs, 0.0, agent_start_pos=[(1, 8)])
            Environment.evaluate_agent(grid, [agent], 1000, out_runs, 0.0, agent_start_pos=[(8, 1)])
            Environment.evaluate_agent(grid, [agent], 1000, out_runs, 0.0, agent_start_pos=[(8, 8)])

            Environment.evaluate_agent(grid, [agent], 1000, out_runs, 0.25, agent_start_pos=[(1, 1)])
            Environment.evaluate_agent(grid, [agent], 1000, out_runs, 0.25, agent_start_pos=[(1, 8)])
            Environment.evaluate_agent(grid, [agent], 1000, out_runs, 0.25, agent_start_pos=[(8, 1)])
            Environment.evaluate_agent(grid, [agent], 1000, out_runs, 0.25, agent_start_pos=[(8, 8)])


if __name__ == "__main__":
    main(
        grid_paths=[Path("grid_configs/warehouse_dyn_8.grd")],
        dynamics_fp=Path("dynamic_env_config/test.json"),
        no_gui=False,
        iters=100,
        fps=10,
        sigma=0,
        out_runs=Path("results/"),
        out_experiments=Path("experiments/"),
        random_seed=0
    )
