import numpy as np
from world import grid
from world import Grid
from pathlib import Path

def agent_vision(loc: tuple, grid: Grid, vis_range: int) -> np.ndarray:
    """    
    Determines the vision of the agent from its current location with

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
        quadrant_vision(loc_arr, quad, grid, vision, vis_range)

    return vision

def quadrant_vision(loc_arr: np.ndarray, quadrant: np.ndarray, grid: Grid,
                    vision: np.ndarray, vis_range: int) -> np.ndarray:
    
    center = np.array([vis_range, vis_range])
    for r in range(1, vis_range + 1):
        for snd in range(1, r + 1):
            if r + snd > vis_range:
                break
            true_diag_vis(np.array([r, snd]), quadrant, grid, loc_arr, vision, center)
            true_diag_vis(np.array([snd, r]), quadrant, grid, loc_arr, vision, center)

                
def true_diag_vis(rel_sq: np.ndarray, quadrant: np.ndarray, grid: Grid, loc_arr: np.ndarray, 
                  vision: np.ndarray, center: np.ndarray) -> None:
    
    vis_sq = center + rel_sq * quadrant
    grid_sq = loc_arr + rel_sq * quadrant
    if not coord_in_grid(grid_sq, grid):
        print("Out of grid")
        return None

    n1 = (rel_sq + np.array([0, -1])) * quadrant
    n2 = (rel_sq + np.array([-1, 0])) * quadrant
    n3 = (rel_sq + np.array([-1, -1])) * quadrant

    if two_visible([center + n1, center + n2, center + n3], vision):
        vision[vis_sq[0], vis_sq[1]] = grid.cells[grid_sq[0], grid_sq[1]]

def sub_diag_vis(rel_sq: np.ndarray, quadrant: np.ndarray, grid: Grid, loc_arr: np.ndarray, 
                 vision: np.ndarray, center: np.ndarray):
    vis_sq = center + rel_sq * quadrant
    grid_sq = loc_arr + rel_sq * quadrant

    if not coord_in_grid(grid_sq, grid):
        return None

    if rel_sq[0] > rel_sq[1]:
        n1 = (rel_sq + np.array([-1, -1])) * quadrant
        n2 = (rel_sq + np.array([-1, 0])) * quadrant
    elif rel_sq[0] < rel_sq[1]:
        n1 = (rel_sq + np.array([-1, -1])) * quadrant
        n2 = (rel_sq + np.array([0, -1])) * quadrant
    
    if one_visible([n1 + loc_arr, n2 + loc_arr], grid):    
        vision[vis_sq[0], vis_sq[1]] = grid.cells[grid_sq[0], grid_sq[1]]


def one_visible(cell_list: list, vision: np.ndarray) -> bool:
    return sum(visible(cell_list, vision)) >= 1

def two_visible(cell_list: list, vision: np.ndarray) -> bool:
    return sum(visible(cell_list, vision)) >= 2

def all_visible(cell_list: list, vision: np.ndarray) -> bool:
    return sum(visible(cell_list, vision)) == len(cell_list)

def visible(cell_list: list, vision: np.ndarray) -> list:
    return [vision[cell[0], cell[1]] in [0, 3, 4]
            for cell in cell_list]

def coord_in_grid(coord: np.ndarray, grid: Grid) -> bool:
    return 0 <= coord[0] < grid.n_rows and 0 <= coord[1] < grid.n_cols

if __name__ == "__main__":

    grd = Grid.load_grid_file(Path("grid_configs/simple_grid.grd"))
    # print(grd.cells)

    # location = (1, 1)
    # print(agent_vision(location, grd, 1))
    # print(agent_vision(location, grd, 2))
    # print(agent_vision(location, grd, 3))
    # print(agent_vision(location, grd, 4))

    # print(agent_vision((1,7), grd, 3))
    # print(agent_vision((1,7), grd, 5))

    # print(agent_vision((4,4), grd, 5))

