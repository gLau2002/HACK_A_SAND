"""
Sand physics simulation — top-down view in a Pygame window.
Click or drag to place sand; Space = wet/dry sand, P = pause, C = clear.
Window constantly vibrates; sand levels out over time.
"""
import random
import pygame
import numpy as np

from simulation import create_grid, step, place_sand, clear_grid, SAND, EMPTY

# Window and grid
WINDOW_WIDTH = 1024
WINDOW_HEIGHT = 768
CELL_SIZE = 4
GRID_COLS = WINDOW_WIDTH // CELL_SIZE
GRID_ROWS = WINDOW_HEIGHT // CELL_SIZE

# Colors (top-down view)
BACKGROUND = (30, 30, 35)
SAND_COLOR = (194, 178, 128)

# Simulation
FPS = 60
STEPS_PER_FRAME = 2
BRUSH_RADIUS = 2
SHAKE_RANGE = 4


def pixel_to_grid(px: int, py: int) -> tuple[int, int]:
    """Convert pixel coordinates to grid (row, col)."""
    col = px // CELL_SIZE
    row = py // CELL_SIZE
    return row, col


def render(screen: pygame.Surface, offscreen: pygame.Surface, grid: np.ndarray) -> None:
    """Draw the grid onto offscreen surface (centered with padding for shake), then blit with random shake offset to screen."""
    offscreen.fill(BACKGROUND)
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if grid[r, c] == SAND:
                rect = pygame.Rect(
                    SHAKE_RANGE + c * CELL_SIZE,
                    SHAKE_RANGE + r * CELL_SIZE,
                    CELL_SIZE,
                    CELL_SIZE,
                )
                pygame.draw.rect(offscreen, SAND_COLOR, rect)

    dx = random.randint(-SHAKE_RANGE, SHAKE_RANGE)
    dy = random.randint(-SHAKE_RANGE, SHAKE_RANGE)
    src_rect = pygame.Rect(SHAKE_RANGE + dx, SHAKE_RANGE + dy, WINDOW_WIDTH, WINDOW_HEIGHT)
    screen.blit(offscreen, (0, 0), src_rect)
    pygame.display.flip()


def main() -> None:
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Sand Physics — Top-Down")
    clock = pygame.time.Clock()
    offscreen = pygame.Surface((WINDOW_WIDTH + 2 * SHAKE_RANGE, WINDOW_HEIGHT + 2 * SHAKE_RANGE))

    grid = create_grid(GRID_ROWS, GRID_COLS)
    paused = False
    sand_is_wet = False
    mouse_down = False

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    mouse_down = True
                    row, col = pixel_to_grid(*event.pos)
                    place_sand(grid, row, col, BRUSH_RADIUS)
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    mouse_down = False
            elif event.type == pygame.MOUSEMOTION:
                if mouse_down:
                    row, col = pixel_to_grid(*event.pos)
                    place_sand(grid, row, col, BRUSH_RADIUS)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    sand_is_wet = not sand_is_wet
                elif event.key == pygame.K_p:
                    paused = not paused
                elif event.key == pygame.K_c:
                    clear_grid(grid)

        if not paused:
            for _ in range(STEPS_PER_FRAME):
                grid = step(grid, wet=sand_is_wet)

        mode = "Wet" if sand_is_wet else "Dry"
        pygame.display.set_caption(f"Sand Physics — {mode}")
        render(screen, offscreen, grid)
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()
