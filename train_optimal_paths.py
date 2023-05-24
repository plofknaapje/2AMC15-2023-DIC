"""Train.

Train your RL Agent in this file.
Feel free to modify this file as you need.

In this example training script, we use command line arguments. Feel free to
change this to however you want it to work.
"""
from pathlib import Path
from optimal_path import optimal_path
import pandas as pd


def train(grid_paths: list[Path], out_experiments: Path):
    """Main loop of the program."""
    results = {"room": [], "start_position": [], "path_length": []}
    for grid_name in grid_paths:
        grid = Path(grid_name)
        room_name = grid_name.split("/")[1]

        if room_name == "simple_grid.grd":
            starts = [(1, 1), (8, 1), (1, 8), (8, 8)]
        elif room_name == "multi_room.grd":
            starts = [(1, 1), (8, 1), (1, 8), (8, 8)]
        else:
            raise ValueError("No valid room name!")

        for start_pos in starts:
            optimal = optimal_path(grid, start_pos)
            path_len = optimal[0]
            start_pos = (start_pos[1], start_pos[0])
            results["room"].append(room_name)
            results["start_position"].append(start_pos)
            results["path_length"].append(path_len)

    data = pd.DataFrame(results)
    data.to_csv(out_experiments / "optimal_path.csv", index=False)


if __name__ == '__main__':
    train(
        grid_paths=["grid_configs/simple_grid.grd", "grid_configs/multi_room.grd"],
        out_experiments=Path("experiments/")
    )
