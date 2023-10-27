from typing import Iterable
import numpy as np


def oor_distance(position, oor=0, size=1, normalized=True):
    """Returns shortest distance on a circular chromosome from a position to the OOR:"""
    # Input validation
    # Handle ragged arrays:
    if isinstance(position, list) and np.all([isinstance(x, Iterable) for x in position]):
        print("Warning: Converting list to numpy array")
        p_out = np.nan * np.ones(shape=(len(position), np.max([len(p) for p in position])))
        for i, p in enumerate(position):
            p_out[i, : len(p)] = p
        position = p_out
    else:
        position = np.array(position)
    oor = np.array(oor).flatten()
    size = np.array(size).flatten()

    if position.ndim > 1:
        n_genomes, _ = position.shape
    else:
        n_genomes = 1
    if len(oor) > 1 and len(oor) != n_genomes:
        raise ValueError("oor must be scalar or have same length as positions")
    if len(size) > 1 and len(size) != n_genomes:
        raise ValueError("size must be scalar or have same length as positions")
    if len(oor) > 1 and len(size) > 1 and len(oor) != len(size):
        raise ValueError("oor and size must have same length")

    dists1 = np.abs(position - oor[:, None])
    dists2 = size - dists1
    dists = np.minimum(dists1, dists2)

    return 2 * dists / size[:, None] if normalized else dists
