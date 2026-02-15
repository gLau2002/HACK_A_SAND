"""
Grid-based sand physics simulation.
Uses double-buffering: read from current grid, write to next grid, then swap.
"""
import numpy as np
import random

# Cell types
EMPTY = 0
SAND = 1


def create_grid(rows: int, cols: int) -> np.ndarray:
    """Create an empty 2D grid (rows x cols)."""
    return np.zeros((rows, cols), dtype=np.uint8)


def _is_supported(grid: np.ndarray, r: int, nc: int) -> bool:
    """Return True if cell (r, nc) has solid support below (bottom boundary or non-empty)."""
    rows = grid.shape[0]
    if r + 1 >= rows:
        return True
    return grid[r + 1, nc] != EMPTY


def step(grid: np.ndarray, wet: bool = False) -> np.ndarray:
    """
    Advance the simulation one step. Sand falls down; if blocked (dry only), tries down-left or down-right.
    Dry: then horizontal settling. Wet: straight down only, no diagonal or horizontal move.
    Returns a new grid (double-buffer).
    """
    rows, cols = grid.shape
    next_grid = grid.copy()

    # Pass 1: falling — iterate from bottom to top so lower sand is updated first
    for r in range(rows - 1, -1, -1):
        for c in range(cols):
            if grid[r, c] != SAND:
                continue
            dr = r + 1
            if dr >= rows:
                continue
            if grid[dr, c] == EMPTY and next_grid[dr, c] == EMPTY:
                next_grid[r, c] = EMPTY
                next_grid[dr, c] = SAND
                continue
            # Dry only: try diagonal (down-left / down-right). Wet: grain stays.
            if wet:
                continue
            candidates = [(dr, c - 1), (dr, c + 1)]
            random.shuffle(candidates)
            for nr, nc in candidates:
                if nc < 0 or nc >= cols:
                    continue
                if grid[nr, nc] == EMPTY and next_grid[nr, nc] == EMPTY:
                    next_grid[r, c] = EMPTY
                    next_grid[nr, nc] = SAND
                    break

    if wet:
        return next_grid

    # Pass 2: horizontal settling — sand moves sideways into supported empty cells (dry only)
    settled_grid = next_grid.copy()
    sand_cells = [(r, c) for r in range(rows) for c in range(cols) if next_grid[r, c] == SAND]
    random.shuffle(sand_cells)
    for r, c in sand_cells:
        if settled_grid[r, c] != SAND:
            continue
        candidates = [(r, c - 1), (r, c + 1)]
        random.shuffle(candidates)
        for _, nc in candidates:
            if nc < 0 or nc >= cols:
                continue
            if next_grid[r, nc] == EMPTY and settled_grid[r, nc] == EMPTY and _is_supported(next_grid, r, nc):
                settled_grid[r, c] = EMPTY
                settled_grid[r, nc] = SAND
                break

    return settled_grid


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


def clear_grid(grid: np.ndarray) -> None:
    """Set all cells to empty. Modifies grid in place."""
    grid.fill(EMPTY)
