"""
Grid-based sand physics simulation.
Uses double-buffering: read from current grid, write to next grid, then swap.
"""
import numpy as np
import random

# Cell types
EMPTY = 0
SAND = 1
WATER = 2
WET_SAND = 3

# Absorption limits (performance)
BFS_MAX_DEPTH = 10
ABSORPTION_MAX_WATER_CELLS = 100
WET_SAND_BATCH_SIZE = 10


def create_grid(rows: int, cols: int) -> np.ndarray:
    """Create an empty 2D grid (rows x cols)."""
    return np.zeros((rows, cols), dtype=np.uint8)


def _is_supported(grid: np.ndarray, r: int, nc: int) -> bool:
    """Return True if cell (r, nc) has solid support below (bottom boundary or non-empty)."""
    rows = grid.shape[0]
    if r + 1 >= rows:
        return True
    return grid[r + 1, nc] != EMPTY


def _neighbors4(grid: np.ndarray, r: int, c: int):
    """Yield in-bounds 4-neighbor (r, c) pairs."""
    rows, cols = grid.shape
    for nr, nc in [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]:
        if 0 <= nr < rows and 0 <= nc < cols:
            yield nr, nc


def _has_dry_sand_via_wet_sand(grid: np.ndarray, r: int, c: int, max_depth: int = BFS_MAX_DEPTH) -> bool:
    """From water cell (r,c), BFS through WET_SAND neighbors (depth-limited); return True if any reached cell has a SAND neighbor."""
    stack = []
    for nr, nc in _neighbors4(grid, r, c):
        if grid[nr, nc] == WET_SAND:
            stack.append((nr, nc, 0))
    visited = set()
    while stack:
        cr, cc, depth = stack.pop()
        if (cr, cc) in visited or depth > max_depth:
            continue
        visited.add((cr, cc))
        for nr, nc in _neighbors4(grid, cr, cc):
            if grid[nr, nc] == SAND:
                return True
            if grid[nr, nc] == WET_SAND and (nr, nc) not in visited:
                stack.append((nr, nc, depth + 1))
    return False


def _pick_dry_sand_to_wet(grid: np.ndarray, water_r: int, water_c: int, max_depth: int = BFS_MAX_DEPTH):
    """Return (row, col) of one dry sand to convert to wet: direct SAND neighbor, or SAND adjacent to wet-sand component (depth-limited)."""
    for nr, nc in _neighbors4(grid, water_r, water_c):
        if grid[nr, nc] == SAND:
            return (nr, nc)
    visited = set()
    stack = []
    for nr, nc in _neighbors4(grid, water_r, water_c):
        if grid[nr, nc] == WET_SAND:
            stack.append((nr, nc, 0))
    while stack:
        cr, cc, depth = stack.pop()
        if (cr, cc) in visited or depth > max_depth:
            continue
        visited.add((cr, cc))
        for nr, nc in _neighbors4(grid, cr, cc):
            if grid[nr, nc] == SAND:
                return (nr, nc)
            if grid[nr, nc] == WET_SAND and (nr, nc) not in visited:
                stack.append((nr, nc, depth + 1))
    return None


def _convert_dry_sand_patch(grid: np.ndarray, seed_r: int, seed_c: int, max_cells: int) -> None:
    """BFS from (seed_r, seed_c) over SAND cells; convert up to max_cells to WET_SAND. Modifies grid in place."""
    rows, cols = grid.shape
    if grid[seed_r, seed_c] != SAND:
        return
    converted = 0
    stack = [(seed_r, seed_c)]
    visited = set()
    while stack and converted < max_cells:
        cr, cc = stack.pop()
        if (cr, cc) in visited:
            continue
        visited.add((cr, cc))
        if grid[cr, cc] != SAND:
            continue
        grid[cr, cc] = WET_SAND
        converted += 1
        for nr, nc in _neighbors4(grid, cr, cc):
            if grid[nr, nc] == SAND and (nr, nc) not in visited:
                stack.append((nr, nc))


def step(grid: np.ndarray, run_absorption: bool = True) -> np.ndarray:
    """
    Advance the simulation one step. Sand falls down; if blocked, tries down-left or down-right.
    If run_absorption is True, also run the water-absorption pass (call with True once per frame).
    Returns a new grid (double-buffer).
    """
    rows, cols = grid.shape
    next_grid = grid.copy()

    # Pass 1: sand and wet sand fall — move into EMPTY or swap with WATER (displacement)
    for r in range(rows - 1, -1, -1):
        for c in range(cols):
            cell = next_grid[r, c]
            if cell not in (SAND, WET_SAND):
                continue
            dr = r + 1
            if dr >= rows:
                continue
            below = next_grid[dr, c]
            if below == EMPTY:
                next_grid[r, c] = EMPTY
                next_grid[dr, c] = cell
                continue
            if below == WATER:
                next_grid[r, c] = WATER
                next_grid[dr, c] = cell
                continue
            left_first = random.random() < 0.5
            diag_order = [(dr, c - 1), (dr, c + 1)] if left_first else [(dr, c + 1), (dr, c - 1)]
            for nr, nc in diag_order:
                if nc < 0 or nc >= cols:
                    continue
                nb = next_grid[nr, nc]
                if nb == EMPTY:
                    next_grid[r, c] = EMPTY
                    next_grid[nr, nc] = cell
                    break
                if nb == WATER:
                    next_grid[r, c] = WATER
                    next_grid[nr, nc] = cell
                    break

    # Pass 2: water — fall (down, then diagonals), then spread horizontally
    for r in range(rows - 1, -1, -1):
        for c in range(cols):
            if next_grid[r, c] != WATER:
                continue
            dr = r + 1
            if dr >= rows:
                continue
            if next_grid[dr, c] == EMPTY:
                next_grid[r, c] = EMPTY
                next_grid[dr, c] = WATER
                continue
            left_first = random.random() < 0.5
            diag_order = [(dr, c - 1), (dr, c + 1)] if left_first else [(dr, c + 1), (dr, c - 1)]
            for nr, nc in diag_order:
                if nc < 0 or nc >= cols:
                    continue
                if next_grid[nr, nc] == EMPTY:
                    next_grid[r, c] = EMPTY
                    next_grid[nr, nc] = WATER
                    break

    # Water horizontal spread: water that didn't fall can move left or right
    water_idx = np.argwhere(next_grid == WATER)
    if water_idx.size > 0:
        water_idx = water_idx[np.random.permutation(water_idx.shape[0])]
        for i in range(water_idx.shape[0]):
            r, c = int(water_idx[i, 0]), int(water_idx[i, 1])
            if next_grid[r, c] != WATER:
                continue
            left_first = random.random() < 0.5
            sides = [(r, c - 1), (r, c + 1)] if left_first else [(r, c + 1), (r, c - 1)]
            for nr, nc in sides:
                if nc < 0 or nc >= cols:
                    continue
                if next_grid[nr, nc] == EMPTY:
                    next_grid[r, c] = EMPTY
                    next_grid[nr, nc] = WATER
                    break

    if run_absorption:
        # Pass 3: absorption — water adjacent to dry sand or wet-sand-connected-to-dry becomes EMPTY, one SAND becomes WET_SAND
        water_idx = np.argwhere(next_grid == WATER)
        if water_idx.size > 0:
            perm = np.random.permutation(water_idx.shape[0])
            water_idx = water_idx[perm[:ABSORPTION_MAX_WATER_CELLS]]
            for i in range(water_idx.shape[0]):
                r, c = int(water_idx[i, 0]), int(water_idx[i, 1])
                if next_grid[r, c] != WATER:
                    continue
                dry = None
                for nr, nc in _neighbors4(next_grid, r, c):
                    if next_grid[nr, nc] == SAND:
                        dry = (nr, nc)
                        break
                if dry is not None:
                    next_grid[r, c] = EMPTY
                    _convert_dry_sand_patch(next_grid, dry[0], dry[1], WET_SAND_BATCH_SIZE)
                    continue
                if not _has_dry_sand_via_wet_sand(next_grid, r, c):
                    continue
                dry = _pick_dry_sand_to_wet(next_grid, r, c)
                if dry is not None:
                    next_grid[r, c] = EMPTY
                    _convert_dry_sand_patch(next_grid, dry[0], dry[1], WET_SAND_BATCH_SIZE)

    return next_grid
    # Pass 2: horizontal settling — sand moves sideways into supported empty cells
    # settled_grid = next_grid.copy()
    # sand_cells = [(r, c) for r in range(rows) for c in range(cols) if next_grid[r, c] == SAND]
    # random.shuffle(sand_cells)
    # for r, c in sand_cells:
    #     if settled_grid[r, c] != SAND:
    #         continue
    #     candidates = [(r, c - 1), (r, c + 1)]
    #     random.shuffle(candidates)
    #     for _, nc in candidates:
    #         if nc < 0 or nc >= cols:
    #             continue
    #         if next_grid[r, nc] == EMPTY and settled_grid[r, nc] == EMPTY and _is_supported(next_grid, r, nc):
    #             settled_grid[r, c] = EMPTY
    #             settled_grid[r, nc] = SAND
    #             break

    # return settled_grid


def place_sand(grid: np.ndarray, row: int, col: int, radius: int = 0) -> None:
    """Place sand at (row, col). If radius > 0, fill a circle of that radius (in cells). Modifies grid in place."""
    rows, cols = grid.shape
    if radius <= 0:
        if 0 <= row < rows and 0 <= col < cols:
            grid[row, col] = SAND
        return
    r2 = radius * radius
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            if dr * dr + dc * dc <= r2:
                r, c = row + dr, col + dc
                if 0 <= r < rows and 0 <= c < cols:
                    grid[r, c] = SAND


def place_water(grid: np.ndarray, row: int, col: int, radius: int = 0) -> None:
    """Place water at (row, col). If radius > 0, fill a circle of that radius (in cells). Modifies grid in place."""
    rows, cols = grid.shape
    if radius <= 0:
        if 0 <= row < rows and 0 <= col < cols:
            grid[row, col] = WATER
        return
    r2 = radius * radius
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            if dr * dr + dc * dc <= r2:
                r, c = row + dr, col + dc
                if 0 <= r < rows and 0 <= c < cols:
                    grid[r, c] = WATER


def clear_grid(grid: np.ndarray) -> None:
    """Set all cells to empty. Modifies grid in place."""
    grid.fill(EMPTY)
