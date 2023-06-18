import numpy as np
import itertools as it
import pandas as pd
from collections import deque


def shortest_path(grid_fp: str, start: tuple[int], end: tuple[int]) -> tuple[list, int] | None:
    """
    Get the shortest possible path between two coordinates on a specified grid.

    Args:
        grid_fp (str): path to the grid file.
        start (tuple[int]): starting position coordinates.
        end (tuple[int]): ending position coordinates.

    Returns:
        tuple[list,int]: length of the shortest path and the complete shortest path.
    """

    # Load the grid
    grid = np.load(grid_fp)
    grid = np.flip(grid, axis=1)
    grid = np.rot90(grid)

    # Specify movements and grid dimensions
    movements = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    rows = len(grid)
    cols = len(grid[0])

    # Create a visited array and queue for BFS traversal
    visited = [[False] * cols for _ in range(rows)]
    queue = deque()

    # Add start coordinate to queue
    queue.append(start)
    visited[start[0]][start[1]] = True

    # Dictionary to store parent of each visited coordinate
    parent = {}

    # BFS traversal
    while queue:
        current = queue.popleft()

        if current == end:
            break

        # Explore possible movements from current
        for movement in movements:
            next_row = current[0] + movement[0]
            next_col = current[1] + movement[1]

            # Within bounds?
            if next_row >= 0 and next_row < rows and next_col >= 0 and next_col < cols:
                # Valid and unvisited tile?
                if grid[next_row][next_col] in [0, 3, 4] and not visited[next_row][next_col]:
                    # Add next coordinate to queue and mark as visited
                    queue.append((next_row, next_col))
                    visited[next_row][next_col] = True
                    # Set current coordinate as parent of next coordinate
                    parent[(next_row, next_col)] = current

    # Reconstruct path if valid path exists, and return tuple with number of steps and path
    if end in parent:
        path = []
        current = end
        while current != start:
            path.append(current)
            current = parent[current]
        path.append(start)
        path.reverse()
        return len(path)-1, path

    # If no path found return None
    else:
        return None


def optimal_path(grid_fp: str, start_pos: tuple[int]) -> tuple[int, list]:
    """
    Get the optimal path and the number of steps for a specific grid, starting position and one agent.
    ATTENTION: ONLY WORKS FOR GRIDS WITH MAX ~8 DIRT TILES!

    Args:
        grid_fp (str): path to the grid file.
        start_pos (tuple[int]): starting position coordinates.

    Returns:
        tuple[int, list]: length of the optimal path and the optimal path itself.
    """
    # Load the grid
    grid = np.load(grid_fp)
    grid = np.flip(grid, axis=1)
    grid = np.rot90(grid)
    # print('GRID:\n', grid, '\n')

    # Define the start, end and dirt tile coordinates
    start = start_pos
    end = tuple(np.argwhere(grid == 4)[0])
    dirts = np.argwhere(grid == 3)

    # Compute all possible permutations of dirt tiles order
    permutationz = list(it.permutations(dirts))
    # Restructure list of permutations into list of lists of coordinate tuples
    permutations = []
    for permutation in permutationz:
        lst = []
        for i in range(len(permutation)):
            tpl = tuple(permutation[i])
            lst.append(tpl)
        permutations.append(lst)

    # Create a graph with the distances between any two checkpoints
    checkpoints = [start] + permutations[0] + [end]
    graph = pd.DataFrame(columns=checkpoints, index=checkpoints)
    graph.iloc[:] = 0

    i = 0
    for p1 in checkpoints:
        j = 0
        for p2 in checkpoints:
            if i < j:
                dist = shortest_path(grid_fp, p1, p2)[0]
                graph[p1][p2] = dist
                graph[p2][p1] = dist
            j += 1
        i += 1
    print('CHECKPOINT DISTANCES GRAPH:\n', graph, '\n')

    # Initialize optimal path and length values
    optimal_path = None
    optimal_length = float('inf')

    # For each permutation of checkpoints order compute the length
    for permutation in permutations:
        path = [start] + permutation + [end]

        length = 0
        for i in range(len(path) - 1):
            current_node = path[i]
            next_node = path[i+1]
            length += graph[current_node][next_node]

        # Update the optimal length and path if the permutation is better than current best
        if length < optimal_length:
            optimal_path = path
            optimal_length = length

    # Create a list with the exact coordinates for every step
    optimal_path_steps = []
    for i in range(len(optimal_path)-1):
        optimal_path_steps = optimal_path_steps + shortest_path(grid_fp, optimal_path[i], optimal_path[i+1])[1]

    return optimal_length, optimal_path_steps
