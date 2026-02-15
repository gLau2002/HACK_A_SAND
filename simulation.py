"""
Grid-based sand physics simulation.
Uses double-buffering: read from current grid, write to next grid, then swap.
"""
import math
import random

import numpy as np

# Cell types
EMPTY = 0
SAND = 1
WATER = 2
WET_SAND = 3

# Absorption limits (performance)
BFS_MAX_DEPTH = 10
ABSORPTION_MAX_WATER_CELLS = 100
WET_SAND_BATCH_SIZE = 10

# Wet sand structure: 10% chance to fall diagonally when blocked below (retains shape better)
WET_SAND_DIAGONAL_PROBABILITY = 0.02


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
            # Wet sand rarely falls diagonally — retains structure when blocked below
            if cell == WET_SAND and random.random() >= WET_SAND_DIAGONAL_PROBABILITY:
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


def erase_region(
    grid: np.ndarray,
    row_min: int,
    row_max: int,
    col_min: int,
    col_max: int,
) -> None:
    """Set all cells in the given rectangle (inclusive) to EMPTY. Clamps to grid bounds. Modifies grid in place."""
    rows, cols = grid.shape
    r0 = max(0, min(row_min, row_max))
    r1 = min(rows - 1, max(row_min, row_max))
    c0 = max(0, min(col_min, col_max))
    c1 = min(cols - 1, max(col_min, col_max))
    grid[r0 : r1 + 1, c0 : c1 + 1] = EMPTY


def clear_grid(grid: np.ndarray) -> None:
    """Set all cells to empty. Modifies grid in place."""
    grid.fill(EMPTY)


def apply_tornado(grid: np.ndarray, center_col: int, radius: int = 8, intensity: float = 0.4) -> None:
    """
    Vortex displacement: non-EMPTY cells in a circular region get rotated around
    the center (horizontal fling) and lifted upward. Modifies grid in place.
    """
    rows, cols = grid.shape
    center_row = rows - 1
    r2 = radius * radius
    moves = []
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            if dr * dr + dc * dc > r2:
                continue
            r, c = center_row + dr, center_col + dc
            if 0 <= r < rows and 0 <= c < cols and grid[r, c] != EMPTY:
                dist_from_center = math.sqrt(dr * dr + dc * dc)
                if dist_from_center < 0.5:
                    continue
                angle = math.atan2(dc, -dr)
                new_angle = angle + intensity * math.pi
                lift = int(dist_from_center * intensity * 2)
                fling = 1.5
                new_dr = -dist_from_center * math.cos(new_angle) * fling
                new_dc = dist_from_center * math.sin(new_angle) * fling
                nr = max(0, min(rows - 1, int(center_row + new_dr - lift)))
                nc = max(0, min(cols - 1, int(center_col + new_dc)))
                moves.append((r, c, nr, nc, int(grid[r, c])))
    for r, c, _, _, _ in moves:
        grid[r, c] = EMPTY
    for _, _, nr, nc, cell in moves:
        grid[nr, nc] = cell


def apply_earthquake(grid: np.ndarray, intensity: float = 0.15) -> None:
    """
    Local chaotic shuffling: non-EMPTY cells swap with a random 4-neighbor with probability p.
    Modifies grid in place.
    """
    rows, cols = grid.shape
    non_empty = [(r, c) for r in range(rows) for c in range(cols) if grid[r, c] != EMPTY]
    random.shuffle(non_empty)
    for r, c in non_empty:
        if random.random() >= intensity:
            continue
        candidates = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
        candidates = [(nr, nc) for nr, nc in candidates if 0 <= nr < rows and 0 <= nc < cols]
        if not candidates:
            continue
        nr, nc = random.choice(candidates)
        grid[r, c], grid[nr, nc] = grid[nr, nc], grid[r, c]


def apply_tsunami(
    grid: np.ndarray,
    side: str,
    wave_height: int = 8,
    wave_width: int = 2,
) -> None:
    """
    Spawn a hump of WATER from the chosen side (left or right). Vertical hump profile:
    more water in middle rows, less at top/bottom. Displaces SAND/WET_SAND inward
    when placing water. Modifies grid in place.
    """
    rows, cols = grid.shape
    if cols < wave_width or rows < 2:
        return
    mid = (rows - 1) / 2.0
    half_range = rows / 2.0
    # Vertical hump: profile[r] = 1 at center, 0 at top/bottom
    profile = np.zeros(rows)
    for r in range(rows):
        t = (r - mid) / half_range if half_range > 0 else 0
        profile[r] = max(0.0, 1.0 - t * t)

    if side == "left":
        col_range = range(0, wave_width)
    else:
        col_range = range(cols - 1, cols - wave_width - 1, -1)

    for r in range(rows):
        if profile[r] <= 0:
            continue
        for c in col_range:
            if grid[r, c] == WATER:
                continue
            if grid[r, c] in (SAND, WET_SAND):
                if side == "left":
                    # Shift row rightward: free (r,c) by pushing content right
                    saved = grid[r, c]
                    for c2 in range(c, cols - 1):
                        grid[r, c2] = grid[r, c2 + 1]
                    grid[r, cols - 1] = saved
                else:
                    # Shift row leftward: free (r,c) by pushing content left
                    saved = grid[r, c]
                    for c2 in range(c, 0, -1):
                        grid[r, c2] = grid[r, c2 - 1]
                    grid[r, 0] = saved
            grid[r, c] = WATER
