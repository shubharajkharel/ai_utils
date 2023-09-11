import matplotlib.pyplot as plt
import numpy as np
from typing import List

def plot_keys(data, keys: List = ["pulse", "no_pulse", "pulse_all"], rows=3, cols=1, **kwargs):
    plot_data = {}
    plot_data["values"] = [data[key] for key in keys]
    plot_data["titles"] = keys
    fig = grid_plot(plot_data, rows, cols, **kwargs)
    return fig

def grid_plot(data, rows_count=1, cols_count=1,filename="grid_plot"):
    num_plots = len(data["values"])

    # add more subplots if there are more data than could be fit in grid
    plot_count = rows_count * cols_count
    if plot_count < num_plots:
        rows_count = int(np.ceil(np.sqrt(num_plots)))
        cols_count = rows_count

    # create the grid of subplots
    fig, axes = plt.subplots(rows_count, cols_count, figsize=(12, 8))

    axes = axes.flatten()

    for i in range(num_plots):
        # calculate subplot position
        row = i % rows_count
        col = i // rows_count

        axes_index = col * rows_count + row

        new_title = data["titles"][i] + " (sample: " + str(len(data["values"][i])) + ")"

        # print(row, cols, axes_index, axes)

        # get the current subplot axis
        ax = axes[axes_index]

        # plot the average waveform on the current subplot axis
        create_subplot(data["values"][i], ax=ax, title=new_title)

    # adjust the spacing between subplots
    plt.tight_layout()

    # # save the figure
    fig.savefig(filename + ".png")

    fig.show()

    return fig


def create_subplot(data, ax=None, title="Average"):
    data_avg = np.mean(data, axis=0)
    data_std = np.std(data, axis=0)

    # x-axis values (assuming each sample is equally spaced)
    x = np.arange(len(data_avg))

    # plot the average waveform on the given axis or create a new one
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(x, data_avg, label="Average")
    ax.fill_between(
        x,
        data_avg - data_std,
        data_avg + data_std,
        alpha=0.3,
        label="Standard Deviation",
    )
    ax.set_xlabel("Time")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    ax.legend()

    return ax


if __name__ == "__main__":
    # Generate sample data
    num_samples = 100
    num_channels = 3
    data = np.random.randn(num_channels, num_samples)

    data = {"values": data, "titles": ["1", "2", "3"]}

    # Define the number of rows and columns for the subplots grid
    rows = 1
    cols = 3
    fig = grid_plot(data, rows, cols)

    # Show the figure
    plt.show()



