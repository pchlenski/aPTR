import numpy as np


def oor_distance(position, oor=0, size=1, normalized=True):
    """Returns shortest distance on a circular chromosome from a position to the OOR:"""
    # Input validation:
    position = np.array(position)
    oor = float(oor)
    size = float(size)

    # Shortest distance is inside linear part:
    dists1 = np.abs(position - oor)
    # Shortest distance wraps around:
    dists2 = np.abs(position + size - oor)
    dists3 = np.abs(position - (oor + size))
    dists = np.vstack((dists1, dists2, dists3)).min(axis=0)
    if normalized:
        return 2 * dists / size
    else:
        return dists
