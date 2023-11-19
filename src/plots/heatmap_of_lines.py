from typing import Literal, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm


def heatmap_of_lines(
    data: np.ndarray,
    ax=None,
    height: Union[Literal["same"], int] = "same",
    cmap="jet",
    norm: matplotlib.colors.Normalize = LogNorm(),
    aspect=0.618,  # golden ratio
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

    # compute heatmap values
    temp_data = (data - data.min()) / (data.max() - data.min())
    x_idx = np.floor(temp_data * width).astype(int) - 1
    y_idx = np.array([np.arange(width)] * temp_data.shape[0]) - 1
    for i, j in zip(x_idx, y_idx):
        heatmap[i, j] += 1
    colorbar = ax.imshow(
        heatmap,
        origin="lower",
        cmap=cmap,
        norm=norm,
        aspect=aspect,
    )
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
