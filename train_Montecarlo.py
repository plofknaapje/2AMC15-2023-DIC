"""Train.

Train your RL Agent in this file.
Feel free to modify this file as you need.

In this example training script, we use command line arguments. Feel free to
change this to however you want it to work.
"""
from argparse import ArgumentParser
from pathlib import Path
import pandas as pd
from tqdm import trange

try:
    from agents.monte_carlo_agent import MCAgent
    from world import Environment
except ModuleNotFoundError:
    import sys
    from os import pardir, path

    root_path = path.abspath(
        path.join(path.join(path.abspath(__file__), pardir), pardir)
    )

    if root_path not in sys.path:
        sys.path.extend(root_path)

    from agents.monte_carlo_agent import MCAgent
    from world import Environment


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
    p.add_argument("--iter", type=int, default=2000,
                   help="Number of iterations to go through.")
    p.add_argument("--random_seed", type=int, default=0,
                   help="Random seed value for the environment.")
    p.add_argument("--out", type=Path, default=Path("results/"),
                   help="Where to save training results.")

    return p.parse_args()


def main(
    grid_paths: list[Path],
    no_gui: bool,
    iters: int,
    fps: int,
    sigma: float,
    out: Path,
    random_seed: int,
):
    """Main loop of the program."""
    results=[]
    for grid in grid_paths:
        
        
        # Set up the environment and reset it to its initial state
        env = Environment(
            grid,
            no_gui,
            n_agents=1,
            sigma=sigma,
            target_fps=fps,
            random_seed=random_seed,
            reward_fn='custom'
        )
        obs, info = env.get_observation()

        # Set up the agents from scratch for every grid
        # Add your agents here
        agents = [
            MCAgent(0, obs, gamma=0.75, epsilon = 0.4),
            MCAgent(0, obs, gamma=0.75, epsilon = 0.6),
            MCAgent(0, obs, gamma=0.75, epsilon = 0.5)
        ]

        # Iterate through each agent for `iters` iterations
        total_iterations=100

        for agent in agents:
            #training loops
            for i in range(total_iterations):
                for _ in trange(iters):
                    # Agent takes an action based on the latest observation and info
                    action = agent.take_action(obs, info)

                    # The action is performed in the environment
                    obs, reward, terminated, info = env.step([action])

                    agent.process_reward(action,reward)
                    # If the agent is terminated, we reset the env.
                    if terminated:
                        break
                    agent.process_reward(action, reward)
                try:
                    obs, info, world_stats = env.reset(agent_start_pos=[(i%env.grid.n_cols,_%env.grid.n_rows)])
                except ValueError:
                    continue #this can be made more elegant for the second assignment
                print(world_stats)
            
            info['iteration'] = 0
            Environment.evaluate_agent(grid, [agent], 500, out, 0.2)

            print("")
            print(str(agent))
            stats={}
            startpoints=[(1, 1), (8, 1), (1, 8), (8, 8)]
            for start in startpoints:
                for sigma in [0.0, 0.25]:
                    print(f"{agent}, start={start}, sigma={sigma}")
                    world_stats = Environment.evaluate_agent(
                        grid_fp=grid, agents=[agent], max_steps=1000,  sigma=sigma, out_dir=out,
                        agent_start_pos=[start], random_seed=0)
                    stats["start"] = start
                    stats["agent"] = str(agent)
                    stats["room_name"]=str(grid)
                    stats["sigma"] = sigma
                    stats["gamma"] = agent.gamma
                    stats["epsilon"] = agent.epsilon
                    stats["dirt_cleaned"] = world_stats["total_dirt_cleaned"]
                    results.append(stats)

    results = pd.DataFrame.from_records(results)
    results.to_csv("MC_results.csv", index=False)       

if __name__ == "__main__":
    args = parse_args()
    main(
        args.GRID,
        args.no_gui,
        args.iter,
        args.fps,
        args.sigma,
        args.out,
        args.random_seed,
    )
