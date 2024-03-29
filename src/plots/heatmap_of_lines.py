from typing import Literal, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, NoNorm, PowerNorm


def heatmap_of_lines(
    data: np.ndarray,
    ax=None,
    height: Union[Literal["same"], int] = "same",
    cmap="jet",
    norm: matplotlib.colors.Normalize = LogNorm(),
    aspect=0.618,  # golden ratio
    x_ticks=None,
    y_ticks=None,
):
    """
    Generate a heatmap from multiple lines of data.
    """

    if ax is None:
        ax = plt.gca()

    # initialize heatmap to zeros
    width = data.shape[1]
    height = width if height == "same" else height

    heatmap = np.zeros((width, height))
    max_val = data.max()
    max_val *= 1.1  # add some padding
    for l in data:
        for x_idx, y_val in enumerate(l):
            y_idx = y_val / max_val * height
            y_idx = y_idx.astype(int)
            y_idx = np.clip(y_idx, 0, height - 1)
            heatmap[y_idx, x_idx] += 1
    colorbar = ax.imshow(
        heatmap,
        origin="lower",
        cmap=cmap,
        norm=norm,
        aspect=aspect,
    )

    if x_ticks is not None:
        x_ticks_pos = np.linspace(0, width, len(x_ticks))
        colorbar.axes.xaxis.set_ticks(x_ticks_pos, x_ticks)
    if y_ticks is not None:
        y_ticks_pos = np.linspace(0, height, len(y_ticks))
        colorbar.axes.yaxis.set_ticks(y_ticks_pos, y_ticks)

    return colorbar


if __name__ == "__main__":
    import numpy as np
    from matplotlib.colors import LogNorm

    signal = np.sin(np.linspace(0, 2 * np.pi, 100))
    data_2d = np.array([signal + np.random.normal(size=100) for _ in range(100)])

    # fig = heatmap_of_lines(data=data_2d)
    # fig.savefig("test.png", dpi=300)

    fig, ax = plt.subplots()
    clb = heatmap_of_lines(data=data_2d, ax=ax, norm=None)
    ax.set_xlabel("indices")
    ax.set_ylabel("values")
    plt.colorbar(clb, ax=ax, label="count", orientation="horizontal", pad=0.08)
    plt.show()
