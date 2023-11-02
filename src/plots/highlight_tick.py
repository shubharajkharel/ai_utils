import numpy as np
import matplotlib.pyplot as plt


def highlight_tick(
    ax,
    highlight,
    axis="x",
    precision=2,
):
    # Determine the original number of ticks if not provided
    original_ticks = ax.get_xticks() if axis == "x" else ax.get_yticks()

    # Get the current axis limits
    min_tick, max_tick = ax.get_xlim() if axis == "x" else ax.get_ylim()

    # Generate new ticks
    num_ticks = len(original_ticks)
    new_ticks = np.linspace(min_tick, max_tick, num_ticks)

    # filter out the ticks that are too close to the highlight
    min_dist = original_ticks[1] - original_ticks[0]
    new_ticks = new_ticks[np.abs(new_ticks - highlight) >= min_dist]

    # Include highlight if not present
    if not np.isclose(new_ticks, highlight).any():
        new_ticks = np.append(new_ticks, highlight)
        new_ticks.sort()

    # Set the ticks
    setter = ax.set_xticks if axis == "x" else ax.set_yticks
    setter(new_ticks)

    # Set the tick labels
    label_formatter = ax.set_xticklabels if axis == "x" else ax.set_yticklabels
    tick_labels = [f"{t:.{precision}f} " for t in new_ticks]
    label_formatter(tick_labels)

    # Highlight the specified tick label
    labels = ax.get_xticklabels() if axis == "x" else ax.get_yticklabels()
    for label in labels:
        if label.get_text() == f"{highlight:.{precision}f} ":
            label.set_color("red")
            break


# Example usage of the function
if __name__ == "__main__":
    # Sample data
    mse = np.random.rand(100)

    # Mean of the data to highlight
    mse_mean = mse.mean()

    # Create a plot
    fig, ax = plt.subplots()

    # Plot MSE as a histogram
    ax.hist(mse, bins=30, color="blue", alpha=0.7)

    # Highlight the mean of the MSE on the x-axis
    highlight_tick(ax, highlight=mse_mean)

    # Display the plot
    plt.show()
