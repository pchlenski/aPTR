""" Scripts for solving an OTU table from a DB """

import pickle

def load_table(path):
    """ Load a VSEARCH output table """
    raise NotImplementedError()

def find_candidates(table, db):
    raise NotImplementedError()

def solve_sequences(candidates):
    raise NotImplementedError()

def solve_all(path, db_path, true_values=None):
    db = pd.read_pickle(db_path)
    table = load_table(path)
    candidates = find_candidates(table, db)
    solutions = solve_sequences(candidates)

    # TODO: append true values

    return solutions