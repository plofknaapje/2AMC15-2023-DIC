"""Optimal paths
Calculates the optimal paths from the given start, through all the dirt to the charger for the given grids.
"""
from pathlib import Path
from world.optimal_path import optimal_path
import pandas as pd


def main(grid_paths: list[str], out_experiments: Path):
    """Main loop of the program."""
    results = {"room": [], "start_position": [], "path_length": []}
    for grid in grid_paths:
        room_name = grid.name

        start_pos = (2,2)

        optimal = optimal_path(grid, start_pos)
        path_len = optimal[0]
        start_pos = (start_pos[1], start_pos[0])
        results["room"].append(room_name)
        results["start_position"].append(start_pos)
        results["path_length"].append(path_len)

    data = pd.DataFrame(results)
    data.to_csv(out_experiments / "optimal_path.csv", index=False)


if __name__ == '__main__':
    main(grid_paths=[Path("grid_configs/warehouse_stat_3.grd"), Path("grid_configs/warehouse_stat_5.grd")],
        out_experiments=Path("experiments/"))
