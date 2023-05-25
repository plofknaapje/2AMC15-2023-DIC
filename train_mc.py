"""Train.

Train your RL Agent in this file.
Feel free to modify this file as you need.

In this example training script, we use command line arguments. Feel free to
change this to however you want it to work.
"""
from pathlib import Path

from tqdm import trange
import pandas as pd

try:
    from agents.mc_agent import MCAgent
    from world import Environment
    from world.grid import Grid
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

def reward_func(grid: Grid, info: dict) -> float:
    reward = 0
    if not info["agent_moved"][0] and not info["agent_charging"][0]:
        reward -= 1
    if info["agent_charging"][0]:
        reward += 100
    if info["dirt_cleaned"][0] != 0:
        reward += 10

    return reward


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
    """Main loop of the program."""
    results = []
    for grid in grid_paths:
        room_name = grid.name

        # Set up the environment and reset it to its initial state
        env = Environment(
            grid,
            no_gui,
            n_agents=1,
            agent_start_pos=None,
            sigma=sigma,
            reward_fn=reward_func,
            target_fps=fps,
            random_seed=random_seed,
        )
        obs, info = env.get_observation()

        # Set up the agents from scratch for every grid
        # Add your agents here
        agents = [
            MCAgent(0, gamma=0.9, obs=obs),
            MCAgent(0, gamma=0.6, obs=obs)
        ]

        # Iterate through each agent for `iters` iterations
        for agent in agents:
            cols, rows = obs.shape

            empty_spaces = [
                (i, j) for i in range(cols) for j in range(rows)
                if obs[i, j] == 0
            ]
            for start in empty_spaces:
                obs, info, world_stats = env.reset(agent_start_pos=[start])
                for _ in trange(100):
                    # Agent takes an action based on the latest observation and info
                    action = agent.take_action(obs, info)
                    # The action is performed in the environment
                    obs, reward, terminated, info = env.step([action])

                    agent.add_reward(obs, info)
                    # If the agent is terminated, we reset the env.
                    if terminated:
                        break
                obs, info, world_stats = env.reset()
                agent.process_reward(obs, reward)

            for _ in trange(iters):
                for _ in trange(100):
                    # Agent takes an action based on the latest observation and info
                    action = agent.take_action(obs, info)

                    # The action is performed in the environment
                    obs, reward, terminated, info = env.step([action])

                    agent.add_reward(obs, info)
                    # If the agent is terminated, we reset the env.
                    if terminated:
                        break
                obs, info, world_stats = env.reset()
                agent.process_reward(obs, reward)

            # Only go here AFTER training
            print("")
            print(str(agent))
            # Add starting spaces here
            if room_name == "simple_grid.grd":
                starts = [(1, 1), (8, 1), (1, 8), (8, 8)]
            elif room_name == "multi_room.grd":
                starts = [(1, 1), (8, 1), (1, 8), (8, 8)]
            else:
                raise ValueError("No valid room name!")

            print(agent.state_coverage())

            for start in starts:
                for sigma in [0.0, 0.25]:
                    print(f"{agent}, start={start}, sigma={sigma}")
                    world_stats = Environment.evaluate_agent(
                        grid, [agent], 1000, out_runs, sigma,
                        agent_start_pos=[start], random_seed=0)
                    world_stats["start"] = start
                    world_stats["agent"] = str(agent)
                    world_stats["room"] = room_name
                    world_stats["sigma"] = sigma
                    results.append(world_stats)
                    agent.reset_agent_state()

    results = pd.DataFrame.from_records(results)
    results.to_csv(out_experiments / "monte_carlo_results.csv", index=False)


if __name__ == "__main__":
    train(
        grid_paths=[Path("grid_configs/multi_room.grd")],
                    # Path("grid_configs/multi_room.grd")]
        no_gui=True,
        iters=1000,
        fps=30,
        sigma=0.0,
        out_runs=Path("results/"),
        out_experiments=Path("experiments/"),
        random_seed=0,
    )
