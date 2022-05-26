""" Scripts for solving an OTU table from a DB """

import pandas as pd
import numpy as np
from collections import defaultdict
from .matrix_solver import OTUSolver
from .database import RnaDB



def load_table(path):
    """ Load a VSEARCH output table """

    return pd.read_table(path, index_col=0)



# def find_genomes_by_md5(md5, db):
#     """ Given an md5-hashed seq, return all genome IDs with that sequence """

#     return db[db["md5"] == md5]["genome"].unique()



def find_candidates(sample, db):
    """ For a sample, find sequences that are worth looking at"""

    # Pass 1: find all genomes with nonzero read counts
    md5s = sample[sample > 0].index
    # genome_hits = [(find_genomes_by_md5(md5, db)) for md5 in md5s]
    genome_hits = [db.find_genomes_by_md5(md5) for md5 in md5s]
    counts = defaultdict(lambda: 0)

    # Pass 2: find all genomes with more than 1 md5 matching
    for hit in genome_hits:
        for genome in hit:
            counts[genome] += 1
    genomes_filtered = [key for key in counts if counts[key] > 1]

    # Pass 3: filter to correct number
    candidates = []
    keep_md5s = set()
    for genome in genomes_filtered:
        # genome_md5s = set(db[db["genome"] == genome]["md5"].unique())
        genome_md5s = set(db[genome]["md5"].unique())
        n_seqs = len(genome_md5s)
        if n_seqs == counts[genome]:
            candidates.append(genome)
            keep_md5s = keep_md5s | genome_md5s

    # Filter tabl
    filtered_index = [i for i in sample.index if i in keep_md5s]

    return sample.loc[filtered_index], candidates



def solve_sequences(genome, sample):
    """ Given some genomes IDs and a sample of coverages, estimate abundances/PTRs """

    genomes = {} # TODO: make genomes from db
    coverages = [] # TODO: make coverages from candidate
    solver = OTUSolver(genomes=genomes, coverages=coverages)
    solver.train()

    return solver.ptrs, solver.abundances



def solve_all(path, db_path=None, left_primer=None, right_primer=None, true_values=None):
    """ Calls all the other functions to solve a TSV of coverages with a database """

    # db = pd.read_pickle(db_path)
    if db_path is not None:
        db = RnaDB(load=db_path)
    else:
        db = RnaDB(left_primer=left_primer, right_primer=right_primer)
    table = load_table(path)

    out = pd.DataFrame(columns=["sample", "genome", "ptr"])
    for column in table.columns:
        sample = table[column]
        coverages, candidates = find_candidates(sample, db)
        if len(candidates) > 0:
            print(column)
            print(candidates)
            genomes, all_seqs = db.generate_genome_objects(candidates)
            sample = sample.loc[all_seqs] # Reorder according to generate_genome_objects
            solver = OTUSolver(genomes, coverages=coverages.values)
            solver.train(lr=.001, tolerance=.0001, verbose=True)
            print("Abundances", np.exp(solver.a_hat), "PTRs", np.exp(solver.b_hat))
            print("True", solver.coverages, "Predicted", solver.compute_coverages(solver.a_hat, solver.b_hat))
            print()
        # solutions = solve_sequences(candidates)
        # # TODO: append outputs
        # for genome, ptr in solutions:
        #     out = out.append({"sample" : column, "genome" : genome, "ptr" : ptr}, ignore_index=True)
        # # TODO: append true values

    return out
