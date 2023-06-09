import numpy as np
from grid import Grid
from pathlib import Path

def agent_vision(loc: tuple, grid: Grid, vis_range: int) -> np.ndarray:
    center = np.array([vis_range, vis_range])

    vision = np.full(
        shape=(vis_range * 2 + 1, vis_range * 2 + 1),
        fill_value=2)
    vision[tuple(center)] = grid.cells[loc]
    print(vision)

    # Straight lines from center
    directions = [np.array([0, 1]), np.array([0, -1]),
                  np.array([1, 0]), np.array([-1, 0])]
    for dir in directions:
        for r in range(1, vis_range + 1):
            vis_square = tuple(center + r * dir)
            grid_square = tuple(np.array(loc) + r * dir)
            if grid.cells[grid_square] in [0, 3, 4]:
                vision[vis_square] = grid.cells[grid_square]
            else:
                break

    # Diagonals
    diagonals = [np.array([-1, -1]), np.array([1, -1]),
                 np.array([-1, 1]), np.array([1, 1])]
    for diag in diagonals:
        n1 = tuple(np.array(loc) + np.array([diag[0], 0]))
        n2 = tuple(np.array(loc) + np.array([0, diag[1]]))
        if not (grid.cells[n1] in [0, 3, 4]
                and grid.cells[n2] in [0, 3, 4]):
            continue
        grid_square = tuple(np.array(loc) + diag)
        vis_square = tuple(center + diag)
        vision[vis_square] = grid.cells[grid_square]

        quadrant_vision(loc, diag, grid, vision)

    return vision


def quadrant_vision(loc: tuple, quadrant: np.ndarray, grid: Grid,
                    vision: np.ndarray) -> np.ndarray:
    # Focused on the top right
    relations = [[(2, 1), [(1, 1), (1, 0)]],
                 [(1, 2), [(1, 1), (0, 1)]],
                 [(2, 2), [(2, 1), (1, 2)]]]

    for dir, deps in relations:
        grid_square = np.array(loc) + np.array(dir) * quadrant
        print(grid_square)


grd = Grid.load_grid_file(Path("grid_configs/simple_grid.grd"))
print(grd.cells)

location = (1, 1)
print(agent_vision(location, grd, 3))
