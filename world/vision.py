import numpy as np
from grid import Grid
from pathlib import Path


def agent_vision(loc: tuple, grid: Grid, vis_range: int) -> np.ndarray:
    """
    Determines the vision of the agent from its current location with Manhattan distance.
    A square which is on the horizontal or vertical line is visible if all squares in between are visible.
    The vision in between these lines is determined using quadrant_vision().

    Args:
        loc (tuple): Location of the agent on the grid.
        grid (Grid): Grid on which the agent moves.
        vis_range (int): Vision range of the agent.

    Returns:
        np.ndarray: vision map of the agent using -1 for unseen areas.
    """
    loc_arr = np.array(loc)
    center = np.array([vis_range, vis_range])

    vision = np.full(
        shape=(vis_range * 2 + 1, vis_range * 2 + 1),
        fill_value=-1)
    vision[center[0], center[1]] = grid.cells[loc]

    # Straight lines from center
    directions = [np.array([0, 1]), np.array([0, -1]),
                  np.array([1, 0]), np.array([-1, 0])]
    for dir in directions:
        for r in range(1, vis_range + 1):
            vis_sq = center + r * dir
            grid_sq = loc_arr + r * dir
            if one_visible([vis_sq - dir], vision):
                cell = grid.cells[grid_sq[0], grid_sq[1]]
                if cell in [0, 3, 4]:
                    vision[vis_sq[0], vis_sq[1]] = 0
                else:
                    vision[vis_sq[0], vis_sq[1]] = 2
            else:
                break

    # Diagonals
    quadrants = [np.array([-1, -1]), np.array([1, -1]),
                 np.array([-1, 1]), np.array([1, 1])]
    for quad in quadrants:
        quadrant_vision(loc_arr, quad, grid, vis_range, out=vision)

    return vision


def quadrant_vision(loc_arr: np.ndarray, quadrant: np.ndarray, grid: Grid,
                    vis_range: int, out: np.ndarray):
    """
    Calculates the vision in one of the four quadrants of the vision matrix.
    The calculations are done in-place.

    Args:
        loc_arr (np.ndarray): location of the agent on the grid.
        quadrant (np.ndarray): translation coordinates of the quadrant.
        grid (Grid): original Grid.
        vis_range (int): vision range.
        out (np.ndarray): output vision matrix.
    """

    center = np.array([vis_range, vis_range])
    for r in range(1, vis_range + 1):
        for snd in range(1, r + 1):
            if r + snd > vis_range:
                break
            diag_square_vision(np.array([r, snd]), quadrant, grid, loc_arr, center, out=out)
            diag_square_vision(np.array([snd, r]), quadrant, grid, loc_arr, center, out=out)


def diag_square_vision(rel_sq: np.ndarray, quadrant: np.ndarray, grid: Grid, loc_arr: np.ndarray,
                       center: np.ndarray, out: np.ndarray):
    """
    Calculate the visibility of a single square on the diagonal.
    Results are stored in-place.

    Args:
        rel_sq (np.ndarray): relative square coordinates.
        quadrant (np.ndarray): translation coordinates of the quadrant.
        grid (Grid): original Grid.
        loc_arr (np.ndarray): location of the agent on the grid.
        center (np.ndarray): center coordinates of the vision matrix.
        out (np.ndarray): output vision matrix
    """

    vis_sq = center + rel_sq * quadrant
    grid_sq = loc_arr + rel_sq * quadrant
    if not coord_in_grid(grid_sq, grid):
        print("Out of grid")
        return None

    n1 = (rel_sq + np.array([0, -1])) * quadrant
    n2 = (rel_sq + np.array([-1, 0])) * quadrant
    n3 = (rel_sq + np.array([-1, -1])) * quadrant

    if two_visible([center + n1, center + n2, center + n3], out):
        out[vis_sq[0], vis_sq[1]] = grid.cells[grid_sq[0], grid_sq[1]]


def one_visible(cell_list: list, vision: np.ndarray) -> bool:
    # Helper function to determine if one cell sees the current cell.
    return sum(visible(cell_list, vision)) >= 1


def two_visible(cell_list: list, vision: np.ndarray) -> bool:
    # Helper function to determine if two cells see the current cell.
    return sum(visible(cell_list, vision)) >= 2


def visible(cell_list: list, vision: np.ndarray) -> list:
    # Helper function to determine if a cell is visible.
    return [vision[cell[0], cell[1]] in [0, 3, 4]
            for cell in cell_list]


def coord_in_grid(coord: np.ndarray, grid: Grid) -> bool:
    # Helper function to determine if a coord is valid.
    return 0 <= coord[0] < grid.n_rows and 0 <= coord[1] < grid.n_cols


if __name__ == "__main__":

    grd = Grid.load_grid_file(Path("grid_configs/simple_grid.grd"))
    print(grd.cells)

    location = (1, 1)
    print(agent_vision(location, grd, 1))
    print(agent_vision(location, grd, 2))
    print(agent_vision(location, grd, 3))
    print(agent_vision(location, grd, 4))

    print(agent_vision((1, 7), grd, 3))
    print(agent_vision((1, 7), grd, 5))

    print(agent_vision((4, 4), grd, 5))
