"""
Sand physics simulation — top-down view using matplotlib.
Click or drag to place sand; Space = pause, C = clear.
"""
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation

from simulation import create_grid, step, place_sand, clear_grid, SAND, EMPTY

# Window and grid
WINDOW_WIDTH = 1024
WINDOW_HEIGHT = 768
CELL_SIZE = 4
GRID_COLS = WINDOW_WIDTH // CELL_SIZE
GRID_ROWS = WINDOW_HEIGHT // CELL_SIZE

# Colors (top-down view) — RGB 0–255, normalized for matplotlib
BACKGROUND = (30 / 255, 30 / 255, 35 / 255)
SAND_COLOR = (194 / 255, 178 / 255, 128 / 255)

# Simulation
FPS = 60
STEPS_PER_FRAME = 2
BRUSH_RADIUS = 2


def data_to_grid(ax, xdata: float, ydata: float) -> tuple[int, int] | None:
    """Convert axes data coordinates to grid (row, col). Returns None if out of bounds."""
    if xdata is None or ydata is None:
        return None
    col = int(xdata)
    row = int(ydata)
    if 0 <= row < GRID_ROWS and 0 <= col < GRID_COLS:
        return row, col
    return None


def main() -> None:
    cmap = mcolors.ListedColormap([BACKGROUND, SAND_COLOR])
    norm = mcolors.Normalize(vmin=0, vmax=1)

    fig, ax = plt.subplots(figsize=(WINDOW_WIDTH / 100, WINDOW_HEIGHT / 100), dpi=100)
    fig.canvas.manager.set_window_title("Sand Physics — Top-Down")

    grid = create_grid(GRID_ROWS, GRID_COLS)
    state = type("State", (), {"grid": grid, "paused": False, "mouse_down": False})()

    im = ax.imshow(
        grid,
        cmap=cmap,
        norm=norm,
        extent=[0, GRID_COLS, GRID_ROWS, 0],
        aspect="equal",
        interpolation="nearest",
        origin="upper",
    )
    ax.set_xlim(0, GRID_COLS)
    ax.set_ylim(GRID_ROWS, 0)
    ax.set_axis_off()

    def on_press(event):
        if event.inaxes != ax or event.button != 1:
            return
        state.mouse_down = True
        cell = data_to_grid(ax, event.xdata, event.ydata)
        if cell:
            row, col = cell
            place_sand(state.grid, row, col, BRUSH_RADIUS)

    def on_motion(event):
        if event.inaxes != ax or not state.mouse_down:
            return
        cell = data_to_grid(ax, event.xdata, event.ydata)
        if cell:
            row, col = cell
            place_sand(state.grid, row, col, BRUSH_RADIUS)

    def on_release(event):
        if event.button == 1:
            state.mouse_down = False

    def on_key(event):
        if event.key == " ":
            state.paused = not state.paused
        elif event.key == "c":
            clear_grid(state.grid)

    def animate(_frame):
        if not state.paused:
            for _ in range(STEPS_PER_FRAME):
                state.grid = step(state.grid)
        im.set_data(state.grid)
        return [im]

    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("motion_notify_event", on_motion)
    fig.canvas.mpl_connect("button_release_event", on_release)
    fig.canvas.mpl_connect("key_press_event", on_key)

    _anim = FuncAnimation(
        fig, animate, interval=1000 // FPS, blit=False, cache_frame_data=False
    )
    plt.tight_layout(pad=0)
    plt.show()


if __name__ == "__main__":
    main()
