"""
The solver for the 16S system. This solver is adapted to solve multiply mapped sequences ACROSS genomes.
"""

import numpy as np
import pandas as pd
import warnings
from collections import defaultdict

from scipy.optimize import fsolve

# from .db import RnaDB

def reflect(x: list, oor: float = 0) -> np.array:
    """
    Given a list/array of x-values, reflect them about the origin of replication
    """
    if np.max(x) > 1:
        raise Exception("Normalize x-values first")
    if oor > 1:
        raise Exception("OOR should be between 0 and 1")

    x = np.array(x)
    x = (x - oor) % 1
    x[x>.5] = 1 - x[x>.5]
    return x

def multi_solver_preprocess(x_values_list, mappings_list, oors):
    """
    Preprocessing utility function for multi_solver. Given a list of x values and a list of mappings, create a set of
    inputs for func
    """

    # Check that we have the same number of systems overlapping
    if len(x_values_list) != len(mappings_list):
        raise Exception("'x_values' and 'mappings' lists are not same size")

    # Reflect x-inputs around origin of replication
    # TODO: refactor OOR code into RnaDB.solve_genome()
    if oors:
        x_values_reflected = [reflect(x,y) for x,y in zip(x_values_list, oors)]
    else:
        x_values_reflected = [reflect(x, 0) for x in x_values_list]

    # Elementwise checks on input dimensions. Then, build up inputs
    bins = defaultdict(list)
    # Build up function inputs
    for i, (x_values, mappings) in enumerate(zip(x_values_reflected, mappings_list)):
        l_i = len(x_values)
        m_i = len(mappings)

        # Check some of the length issues we may face
        if l_i != m_i:
            raise Exception("'x_values' and 'mappings' arrays are not the same size")

        # Check x_values is within [0,1):
        if np.max(x_values) > 1:
            raise Exception("max(x_values) > than 1. Please normalize x_values before calling solve_general()")

        # if good, build up an inverse mapping of our constraints
        # Mappings: implicitly maps [x_position => sequence]
        # Inverse mappings: maps [sequence => (list_index, x_position)]
        for j, mapping in enumerate(mappings):
            bins[mapping].append((i, j))

    return x_values_reflected, bins



def multi_solver(
    x_values_list : list,
    mappings_list : list,
    coverages : list,
    oors: list = False,
    m_reg : float = 0,
    b_reg : float = 0,
    initialization : str = 'zero',
    history : bool = False) -> np.array:
    """
    Solve a 16S system of equations.

    This function proceeds in the following phases:
    1.  Run a number of checks on the inputs
    2.  Build up an inverse (aggregate coverage bin --> [16S indices]) dictionary
    3.  Reflect any x-values past the terminus onto the downward phase of the PTR curve
    4.  Loop over number of 16S RNA locations (x/y values) and aggregate coverage bins to build up a system of
        equations representing the appropriate behavior of the Lagrange multipliers
    5.  Use scipy fsolve method to find the roots of this system of equations

    Args:
    -----
    x_values_list:
        List of array-like objects. Each has start positions in the interval [0, 1) for 16S operons. If not
        constrained, we will constrain ourselves.
    mappings_list:
        List of array-like objects, should be same size as x_values. Each should consist of integer from 0 to n, where
        n is the sum of sizes over all elements of 'coverages'. Tells you which coverage a given element of x_values
        contributes to. Note that an index can occur in multiple list elements.
    coverages:
        Array_like, each should be <= x_values size. Observed aggregate coverages from mappings. NOT log-scaled!
    oors:
        List of floats corresponding to the [0,1]-normalized origins of replication.
    m-reg:
        Float. Coefficient of L2 regularization applied to line slope.
    b-reg:
        Float. Coefficient of L2 regularization applied to line intercept.
    iniziatialization:
        String. Should be one of ['zero', 'random', 'one-zero']. Sets the initial y-value guess.
    history:
        Boolean. If true, saves solver history.

    Returns:
    --------
    A vector [ m, b, y1, ..., y_n, c_1, ..., c_n ] of system of equation solutions.

    Raises:
    -------
    TODO
    """

    # Preprocess
    x_values_reflected, bins = multi_solver_preprocess(x_values_list, mappings_list, oors)
    l = len(x_values_reflected)
    n = len(coverages)
    total_x = np.sum(len(x) for x in x_values_reflected)

    # Preprocess X values for numpy
    x_np = [np.array(x) for x in x_values_reflected]

    if n == 1:
        raise Exception("Cannot compute PTR from a single coverage bin")

    # build up our equation
    def func(X, history=history):
        """
        func(x) represents the system of equations we need to solve to retrieve ptr.
        the input x is array-like with the following structure:

            X = < m_1, ..., m_l, b_1, ... b_l, y_11, ... y_1n_1, ... y_l1, ... y_ln_l, lambda_1, ..., lambda_m >

        where:
        * m_i and b_i are slope and intercept of a line of best fit for (x_i:n_i, y_i:n_i)
        * y_i1:n are log-coverage values estimated within the given constraints, and
        * lambda_1:m are lagrange multipliers for our constraints

        func(x) inherits the following variables from solve_general:
        * l                Length of x_values_list, coverages_list, mappings_list
        * total_size        Sum of coverage bin sizes
        * x_values_list     List of x-coordinates in [0,1) reflected about terminus where x > 0.5
        * coverages_list    List of bin coverages
        * mappings_list     List of mappings. For each coverage bin, lists the constituent 16S copy INDICES
        * m_reg             L2 penalty for line slope
        * b_reg             L2 penalty for line intercept
        * bins              Inverse of mappings. at each INDEX, gives a list of (genome index, coverage bin) tuples.
        * history           Object for keeping track of training history
        """

        # Unpack variables from vector
        m_vals = X[:l]
        b_vals = X[l : 2*l]
        y_vals = X[2*l : -n] # Everything between m, b, and lagrange is y-values
        lambdas = X[-n:] # Last n elements are lagrange multipliers

        # More sanity checks: Y inputs
        if len(y_vals) != total_x:
            raise Exception("Numerical error: y_values do not match number of x-values")

        # Reshape Y to match X-values
        # TODO: can this be written more nicely?
        y_np = []
        y_index = 0
        for x in x_values_reflected:
            y_vals_matched = y_vals[y_index : y_index + len(x)]
            y_vals_match = np.array(y_vals_matched)
            y_np.append(y_vals_matched)
            y_index += len(x)

        # Sanity check: all lengths are l
        if len(m_vals) != len(b_vals) != len(y_vals) != len(x_np) != len(y_np) != l:
            raise Exception("Length mismatch somewhere in X_vector unpacking phase")

        # Compute gradient w/r/t each m_i
        m_grads = [np.sum(2*x*(m*x+b-y)) for x,y,m,b in zip(x_np,y_np,m_vals,b_vals)]
        if m_reg:
            m_grads = [dm + 2 * m_reg * m for dm,m in zip(m_grads,m_vals)]

        # Compute gradient w/r/t each b_i
        b_grads = [np.sum(2*(m*x+b-y)) for x,y,m,b in zip(x_np,y_np,m_vals,b_vals)]
        if b_reg:
            b_grads = [db + 2 * b_reg * b for db,b in zip(b_grads,b_vals)]

        # Compute gradients w/r/t each y_i,1 : y_i:j in one shot:
        y_grads = []
        for x, y, m, b, mapping in zip(x_np, y_np, m_vals, b_vals, mappings_list):
            d_y = -2 * m * x + b - y
            h = [lambdas[x] for x in mapping]
            # d_h = np.array(h) * np.log(2) * np.exp2(y)
            d_h = np.array(h) * np.exp(y)
            y_grads += list(d_h-d_y) # += forces y_grads to be 1-D

        # Loop over our discrete coverage bins, retrieve our best guesses for their constituent y-values, and get the
        # gradient of the constraint with respect to it.
        lambda_grads = []
        for coverage_bin in bins:
            # Get coverages, indices, retrieve their values
            coverage_val = coverages[coverage_bin]
            bin_x_vals = bins[coverage_bin]
            bin_y_vals = [y_np[i][j] for i,j in bin_x_vals]

            # Coverage := sum over (i in B) 2^(y_i) = c_i
            coverage_sum = np.sum(np.exp2(bin_y_vals))

            # Difference of actual coverage - coverage bin should be zero
            lambda_grads.append(coverage_sum - coverage_val)

        # Concatenate all outputs into a single vector
        out = [*m_grads, *b_grads, *y_grads, *lambda_grads]
        if history:
            history += [out]
        return out

    # Set params and solve
    if initialization == 'zero':
        initial_ys = [0] * total_x
    elif initialization == 'random':
        initial_ys = .1 * np.random.rand(total_x)
    elif initialization == 'uniform':
        raise NotImplementedError()
        # TODO: implement uniform initialization
    elif initialization == 'one_zero' or initialization == 'one-zero':
        # All coverage goes to leftmost instance
        initial_ys = [np.zeros(len(x)) for x in x_values_reflected]
        for coverage_bin in bins:
            i,j = bins[coverage_bin][0]
            initial_ys[i][j] = np.log2(coverages[coverage_bin] + 1e-5)
        initial_ys = [x for y in initial_ys for x in y]
    else:
        raise Exception(f"initialization method '{initialization}' does not exist!")

    initial_m = [0] * l
    initial_b = [0] * l
    initial_lambdas = [0] * n
    initial_values = [*initial_m, *initial_b, *initial_ys, *initial_lambdas]

    # initialize to lump all coverage to the leftmost points
    results = fsolve(func, initial_values)

    return results
