"""Train.

Train your RL Agent in this file.
Feel free to modify this file as you need.

In this example training script, we use command line arguments. Feel free to
change this to however you want it to work.
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
    out: Path,
    random_seed: int,
):
    """Main loop of the program."""

    for grid in grid_paths:
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
            # NullAgent(0),
            # GreedyAgent(0),
            # RandomAgent(0),
            MCAgent(0, gamma=0.99, obs=obs)
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
                print(info["agent_pos"][0])
                for _ in trange(50):
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

            print(world_stats)
            print(agent.q)
            print(agent.pi)

            Environment.evaluate_agent(grid, [agent], 1000, out, 0.0)
            agent.reset_agent_state()


if __name__ == "__main__":
    train(
        grid_paths=[Path("grid_configs/simple_grid.grd")],
        no_gui=True,
        iters=100,
        fps=30,
        sigma=0.0,
        out=Path("results"),
        random_seed=0,
    )
