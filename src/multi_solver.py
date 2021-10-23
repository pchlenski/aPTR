"""
The solver for the 16S system. This solver is adapted to solve multiply mapped sequences ACROSS genomes.
"""

import numpy as np
import pandas as pd
import warnings
from collections import defaultdict

from scipy.optimize import fsolve

# from .db import RnaDB

def reflect(x: list, oor: float) -> np.array:
    """
    Given a list/array of x-values, reflect them about the origin of replication
    """
    if np.max(x) > 0:
        raise Exception("Normalize x-values first")
    if oor > 1:
        raise Exception("OOR should be between 0 and 1")

    x = x - oor
    x = x % 1
    return x[x>.5] = 1 - x[x>.5]

def multi_solver(
    x_values_list : list,
    mappings_list : list,
    coverages : list,
    oors: list = False,
    regularization : float = 0,
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
        Array_like, each should be <= x_values size. Observed aggregate coverages from mappings.
    oors:
        List of floats corresponding to the [0,1]-normalized origins of replication. 
    regularization:
        Float. Coefficient of L2 regularization applied to line.
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

    # Check that we have the same number of systems overlapping
    l1 = len(x_values_list)
    l2 = len(mappings_list)
    n = len(coverages)
    if l1 != l2:
        raise Exception("'x_values' and 'mappings' lists are not same size")
    total_x = np.sum(len(x) for x in x_values)

    # Check that coverages are well-behaved
    if set(mappings) != set(range(n)):
        raise Exception("entries of 'mapping' are not 0-indexed integers")

    # Reflect x-inputs around origin of replication
    # TODO: refactor OOR code into RnaDB.solve_genome()
    if oors:
        x_values = [reflect(x,y) for x,y in zip(x_values, oors)]
    else:
        x_values = [reflect(x, 0) for x in x_values]

    # Elementwise checks on input dimensions
    # Then, build up inputs
    bins = defaultdict([])
    # Build up function inputs
    for i, x_values, mappings, coverages in enumerate(zip(x_values_list, mappings_list, coverages_list)):
        l = len(x_values[i])
        m = len(mappings[i])
        n = len(coverages[i])

        # Check some of the length issues we may face
        if l != m:
            raise Exception("'x_values' and 'mappings' arrays are not the same size")
        elif n > l:
            raise Exception("'coverages' is larger than 'x_values'")
        elif l == n:
            # warnings.warn("All RNAs map uniquely to a coverage. Computation is trivial")
            pass
        # Simply proceed with the rest
        elif n == 1:
            raise Exception("Cannot compute PTR from a single coverage bin")

        # Check x_values is within [0,1):
        if np.max(x_values) > 1:
            raise Exception("Maximum element of x_values is greater than 1. Please normalize x values by genome length before calling solve_general()")

        # Check that our coverages are well-behaved
        elif len(set(mappings)) > len(coverages):
            raise Exception("'coverages' does not have enough entries for the mapping provided.")
        elif len(set(mappings)) < len(coverages):
            raise Exception("'coverages' has too many entries for the mapping provided.")

        # if good, build up an inverse mapping of our constraints
        # Mappings: implicitly maps [x_position => sequence]
        # Inverse mappings: maps [sequence => (list_index, x_position)]
        for mapping in mappings:
            bins[mapping].append((i, idx))

        # TODO: check that coverages make sense in the context of the mappings
        # i.e. no [coverage / number of operons in bin] should be more than 2x any other

    """
    Old function (single case):
    X = < m, b, y_1, ..., y_n, lambda_1, ..., lambda_m >

    New function (multimap case):
    X = < m_1, ..., m_l, b_1, ... b_l, y_11, ... y_1n_1, ... y_l1, ... y_ln_l, lambda_1, ..., lambda_m >
    Main differences:
      * Fit l different lines simultaneously
      * Coverage constraints go from 2^(y_i) + 2^(y_j) = c_{ij} to 2^{y_ij} + 2^{y_kl} = c_{ijkl}
    """

    # build up our equation
    def func(x, regularization=regularization, history=history): 
        """
        func(x) represents the system of equations we need to solve to retrieve ptr.
        the input x is array-like with the following structure:

            X = < m_1, ..., m_l, b_1, ... b_l, y_11, ... y_1n_1, ... y_l1, ... y_ln_l, lambda_1, ..., lambda_m >

        where:
        * m_i and b_i are slope and intercept of a line of best fit for (x_i:n_i, y_i:n_i)
        * y_i1:n are log-coverage values estimated within the given constraints, and
        * lambda_1:m are lagrange multipliers for our constraints

        func(x) inherits the following variables from solve_general:
        * l1                Length of x_values_list, coverages_list, mappings_list
        * total_size        Sum of coverage bin sizes
        * x_values_list     List of x-coordinates in [0,1) reflected about terminus where x > 0.5
        * coverages_list    List of bin coverages
        * mappings_list     List of mappings. For each coverage bin, lists the constituent 16S copy INDICES
        * bins              Inverse of mappings. at each INDEX, gives a list of (genome index, coverage bin) tuples.
        * history           Object for keeping track of training history
        """

        # Unpack variables from vector
        m_vals = x[0 : l]
        b_vals = x[l : 2*l]
        y_vals = [2*l+1 : -n] # Everything between m, b, and lagrange is y-values
        lambdas = x[-n] # Last n elements are lagrange multipliers

        # More sanity checks: Y inputs
        if len(y_vals) != total_x:
            raise Exception("Numerical error: y_values do not match number of x-values")

        # Preprocess for numpy
        # Reshape y to match v-values
        x_np = [np.array(x) for x in x_values_reflected]
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

        # Compute gradients of m and b
        # OLD FORMULA, 1_D case: dm = np.sum(x_np * (m * x_np + b - y_np))
        dm_vals = [np.sum(x*(m*x+b-y)) for x,y,m,b in zip(x_np,y_np,m_vals,b_vals)]
        dm_vals = [dm + 2 * regularization * m for dm,m in zip(dm_vals,m_vals)]  # L2-regularization
        # OLD FORMULA, 1-D case: db = np.sum(m * x_np + b - y_np)
        db_vals = [np.sum(m*x+b-y) for x,y,m,b in zip(x_np,y_np,m_vals,b_vals)]
        db_vals = [db + 2 * regularization * b for db,b in zip(db_vals,b_vals)] # L2-regularization

        # compute gradients for each yi
        y_grads = []

        for x, y, m, b, mapping in zip(x_values_refleceted, y_values, m_vals, b_vals, mappings):
            y_grads_local = []
            for xi, yi, mi in zip(x, y, mapping):
                """
                Loop over our (x, y) pairs, using the 'mappings' input to work it all out.
                The equation for each y_i is as follows:

                 df                                                y_i
                ---- = -2( mx_i + b - y_i ) = lambda_j * log(2) * 2
                dy_i

                If we let c_j be constraint j, i.e. the aggregate coverage that y_i goes
                into, then the index of c_j is stored in mappings[x_i], hence we can use
                lambdas[m_i] to get the corresponding Lagrange multiplier for it
                """
                dy = -2 * (m * xi + b - yi)
                dlambda = lambdas[mi] * np.log(2) * (2 ** yi)
                y_grads_local.append(dy - dlambda)
            y_grads.append(y_grads_local)

        # compute constraints
        constraint_eqs = []
        for coverage_bin in bins:
            """
            Loop over our discrete coverage bins, retrieve our best guesses for their
            constituent y-values, and get the gradient of the constraint with respect 
            to it. Coverage constraints take the following form:
             __
            \  '          y_i
            /__, i in S  2    = c_i

            TODO: 
            Handle cases where the constraint is simply of the form 2^{y_i} = c_j,
            no summation required
            """
            coverage_val = coverages[coverage_bin] # get actual bin coverage
            bin_x_vals = bins[coverage_bin] # get indices
            bin_y_vals = [y_values[i][j] for i,j in bin_x_vals] # then retrieve their values
            coverage_sum = np.sum(np.exp2(bin_y_vals)) # sum exponentiation to get coverage
            constraint_eqs.append(coverage_sum - coverage_val)

        # concatenate all outputs into a single vector
        out = [*dm_vals, *db_vals, *y_grads, *constraint_eqs]
        # print(out)
        if history:
            history += [out]
        return out

    # set params and solve
    if initialization == 'zero':
        initial_ys = [0] * total_x
    elif initialization == 'random':
        initial_ys = np.random.rand(total_x)
    elif initialization == 'one_zero':
        raise NotImplementedError()
        # TODO: Implement one-zero initialization
    else:
        raise Exception(f"initialization method '{initialization}' does not exist!")
    initial_lambdas = [0] * n
    initial_values = [0, 0, *initial_ys, *initial_lambdas]

    # initialize to lump all coverage to the leftmost points
    results = fsolve(func, initial_values)

    return results
