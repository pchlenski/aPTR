from typing import Iterable
import numpy as np


def _validate_oor_distance_inputs(position, oor, size):
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

    n_genomes = position.shape[1] if position.ndim > 1 else 1
    if len(oor) > 1 and len(oor) != n_genomes:
        raise ValueError("oor must be scalar or have same length as positions")
    if len(size) > 1 and len(size) != n_genomes:
        raise ValueError("size must be scalar or have same length as positions")
    if len(oor) > 1 and len(size) > 1 and len(oor) != len(size):
        raise ValueError("oor and size must have same length")

    return position, oor, size


def oor_distance(position, oor=0, size=1, normalized=True):
    """Returns shortest distance on a circular chromosome from a position to the OOR:"""

    position, oor, size = _validate_oor_distance_inputs(position, oor, size)
    dists = np.minimum(np.abs(position - oor[:, None]), size - dists1)
    return 2 * dists / size[:, None] if normalized else dists
