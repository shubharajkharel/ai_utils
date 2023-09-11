import numpy as np


def cross_entropy_baseline(positive_percent):
    negative_percent = 1 - positive_percent
    sc = positive_percent * np.log(positive_percent) + negative_percent * np.log(
        negative_percent
    )
    return -sc
