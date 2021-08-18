"""
The solver for the 16S system.

NOTE: this is the old version. It cannot handle multiply-mapped sequences ACROSS GENOMES yet.

TODO: handle multiply mapped sequences across genomes.
"""

import numpy as np
import warnings

from scipy.optimize import fsolve

def solver(
    x_values : np.array, 
    mappings : np.array,
    coverages : np.array,
    history : bool = False) -> np.array:
    """
    Solve a 16S system of equations.

    This function proceeds in the following phases:
    1.  Run a number of checks on the inputs
    2.  Build up an inverse (aggregate coverage bin --> [16S indices]) dictionary
    3.  Reflect any x-values past the terminus onto the downward phase of the PTR curve
    4.  Loop over number of 16S RNA locations (x/y values) and aggregate coverage bins to build up a system of equations
        representing the appropriate behavior of the Lagrange multipliers
    5.  Use scipy fsolve method to find the roots of this system of equations

    Args:
    -----
    x_values:
        Array-like. Start positions in the interval [0, 1) for 16S operons. If not constrained, we will constrain 
        ourselves.
    mappings:
        Array-like, should be same size as x_values. Each entry should be an integer from 0 to n, where n is the size
        of 'coverages'. Tells you which coverage a given element of x_values contributes to.
    coverages:
        Array_like, should be <= x_values size. Observed aggregate coverages from mappings.

    Returns:
    --------
    A vector [ m, b, y1, ..., y_n, c_1, ..., c_n ] of system of equation solutions.

    Raises:
    -------
    TODO
    """

    l = len(x_values)  # number of inputs
    m = len(mappings)
    n = len(coverages) # number of constraints

    # check some of the length issues we may face
    if l != m:
        raise Exception("'x_values' and 'mappings' arrays are not the same size")
    elif n > l:
        raise Exception("'coverages' is larger than 'x_values'")
    elif l == n:
        warnings.warn("All RNAs map uniquely to a coverage. Computation is trivial")
    # simply proceed with the rest
    elif n == 1:
        raise Exception("Cannot compute PTR from a single coverage bin")

    # check x_values is within [0,1):
    if np.max(x_values) > 1:
        raise Exception("Maximum element of x_values is greater than 1. Please normalize x values by genome length before calling solve_general()")

    # check that our coverages are well-behaved
    if set(mappings) != set(range(n)):
        raise Exception("entries of 'mapping' are not 0-indexed integers")
    elif len(set(mappings)) > len(coverages):
        raise Exception("'coverages' does not have enough entries for the mapping provided.")
    elif len(set(mappings)) < len(coverages):
        raise Exception("'coverages' has too many entries for the mapping provided.")

    # if good, build up an inverse mapping of our constraints
    bins = { x : [] for x in set(mappings) } # explicitly initialize to none
    for idx in range(m):
        bin = mappings[idx]
        bins[bin] += [idx]
        # print("mapping:\t", bins)

    # TODO: check that coverages make sense in the context of the mappings
    # i.e. no [coverage / number of operons in bin] should be more than 2x any other

    # if good, preprocess x-values to all be on downward phase of PTR:
    x_values_reflected = []
    for x in x_values:
        if x > 0.5:
            x_values_reflected += [1 - x]
        else:
            x_values_reflected += [x]

    # build up our equation
    def func(x, history=history): 
        """
        func(x) represents the system of equations we need to solve to retrieve ptr.
        the input x is array-like with the following structure:

          x = < m, b, y1, ..., y_n, lambda_1, ..., lambda_m >

        where m and b are slope and intercept of a line of best fit for (x_1:n, y_1:n),
        y_1:n are log-coverage values estimated within the given constraints, and
        lambda_1:m are lagrange multipliers for our constraints

        func(x) inherits the following variables from solve_general:
        * x_values_reflected  x-coordinates in [0,1) reflected about terminus where x > 0.5
        * coverages           bin coverages
        * mappings            for each coverage bin, lists the constituent 16S copy INDICES
        * bins                inverse of mappings. at each INDEX, gives the coverage bin.
        * history             object for keeping track of training history
        """

        # unpack variables from x vector
        m        = x[0]
        b        = x[1]
        y_values = x[2:-n] # everything between m, b, and lagrange is y-values
        lambdas  = x[-n:]  # last n elements of x are lagrange multipliers

        # preprocess for numpy
        x_np = np.array(x_values_reflected)
        y_np = np.array(y_values)

        # check that our inputs make sense
        if len(x_np) != len(y_np):
            raise Exception("x and y value arrays are not the same shape")

        # compute gradients of m and b
        dm = np.sum( x_np * (m * x_np + b - y_np) )
        db = np.sum( m * x_np + b - y_np )

        # compute gradients for each yi
        y_grads = []
        for (xi, yi, mi) in zip(x_values_reflected, y_values, mappings):
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
            dy       = -2 * (m * xi + b - yi)
            dlambda  = lambdas[mi] * np.log(2) * (2 ** yi)
            y_grads += [dy - dlambda] 

        # compute constraints
        constraint_eqs = []
        for bin in bins.keys():
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
            coverage_val    = coverages[bin]                          # get actual bin coverage
            bin_x_vals      = bins[bin]                               # get indices
            bin_y_vals      = [ y_values[idx] for idx in bin_x_vals ] # then retrieve their values
            coverage_sum    = np.sum( np.exp2(bin_y_vals) )           # sum exponentiation to get coverage
            constraint_eqs += [coverage_sum - coverage_val]

        # concatenate all outputs into a single vector
        out      = [dm, db, *y_grads, *constraint_eqs]
        # print(out)
        if history != False:
            history += [out]
        return out

    # set params and solve
    initial_ys = [0] * l
    for bin in bins.keys():
        first_idx             = bins[bin][0]            # get leftmost element of bin
        initial_ys[first_idx] = np.log2(coverages[bin]) # give it all the coverage
    initial_lambdas = [0] * n
    initial_values = [0, 0, *initial_ys, *initial_lambdas]
    # print("initial values:\t", initial_values)
    results = fsolve(func, initial_values, history) # initialize to lump all coverage to the leftmost points

    return results