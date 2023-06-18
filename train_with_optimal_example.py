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
    from world import Environment

    # Add your agents here

except ModuleNotFoundError:
    from os import path
    from os import pardir
    import sys

    root_path = path.abspath(path.join(
        path.join(path.abspath(__file__), pardir), pardir)
    )

    if root_path not in sys.path:
        sys.path.extend(root_path)

    from world import Environment

    # Add your agents here
    from agents.null_agent import NullAgent

from optimal_path import optimal_path


def parse_args():
    p = ArgumentParser(description="DIC Reinforcement Learning Trainer.")
    p.add_argument("GRID", type=Path, nargs="+",
                   help="Paths to the grid file to use. There can be more than "
                        "one.")
    p.add_argument("--no_gui", action="store_true",
                   help="Disables rendering to train faster")
    p.add_argument("--sigma", type=float, default=0.,
                   help="Sigma value for the stochasticity of the environment.")
    p.add_argument("--fps", type=int, default=30,
                   help="Frames per second to render at. Only used if "
                        "no_gui is not set.")
    p.add_argument("--iter", type=int, default=100,
                   help="Number of iterations to go through.")
    p.add_argument("--random_seed", type=int, default=0,
                   help="Random seed value for the environment.")
    p.add_argument("--out", type=Path, default=Path("results/"),
                   help="Where to save training results.")

    return p.parse_args()


def main(grid_paths: list[Path], no_gui: bool, iters: int, fps: int,
         sigma: float, out: Path, random_seed: int):
    """Main loop of the program."""

    for grid in grid_paths:

        # Print start position, optimal number of steps and optimal path
        start_pos = (1, 2)                               # Or: info['agent_pos'][0].
        # ^ In that case place this code block after environment intialization.
        print('START POSITION:', start_pos)
        optimal = optimal_path(grid, start_pos)
        print('OPTIMAL NUMBER OF STEPS:', optimal[0])
        print('OPTIMAL PATH:', optimal[1])
        # Invert the start position coordinates so that it matches what we see in the GUI.
        # This way it is consistent with the other functions below.
        start_pos = (start_pos[1], start_pos[0])

        # Set up the environment and reset it to its initial state
        env = Environment(grid, no_gui, n_agents=1, agent_start_pos=[start_pos],
                          sigma=sigma, target_fps=fps, random_seed=random_seed,
                          reward_fn='custom')
        obs, info = env.get_observation()

        agents = [NullAgent(0)]

        # Iterate through each agent for `iters` iterations
        for agent in agents:
            for _ in trange(iters):
                # Agent takes an action based on the latest observation and info
                action = agent.take_action(obs, info)

                # The action is performed in the environment
                obs, reward, terminated, info = env.step([action])

                # If the agent is terminated, we reset the env.
                if terminated:
                    obs, info, world_stats = env.reset()
                agent.process_reward(obs, reward)
            obs, info, world_stats = env.reset()
            print(world_stats)

            Environment.evaluate_agent(grid, [agent], 100, out, 0.2, agent_start_pos=[start_pos])


if __name__ == '__main__':
    args = parse_args()
    main(args.GRID, args.no_gui, args.iter, args.fps, args.sigma, args.out,
         args.random_seed)
