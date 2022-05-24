""" Scripts for solving an OTU table from a DB """

import pandas as pd
from collections import Counter

def load_table(path):
    """ Load a VSEARCH output table """
    return pd.read_table(path, index_col=0)

def find_genomes_by_md5(md5, db):
    """ Given an md5-hashed seq, return all genome IDs with that sequence """
    return set(db[db["md5"] == md5]["genome"])

def find_candidates(table, db):
    """ For a table, find sequences that are worth looking at"""

    # Pass 1: find all genomes
    md5s = table.index
    genome_hits = Counter([find_genomes_by_md5(md5, db) for md5 in md5s])

    # Pass 2: filter to correct number
    candidates = []
    keep_md5s = set()
    for genome in genome_hits:
        genome_md5s = db[db["genome" == genome]]["md5"].unique()
        n_seqs = len(genome_md5s)
        if n_seqs == genome_hits["genome"]:
            candidates.append(genome)
            keep_md5s = keep_md5s | genome_md5s

    # Filter tabl
    filtered_index = [i for i in table.index if i in keep_md5s]

    return table[filtered_index]


# def hits_exist(otus, table=t):
#     overlaps, seqs = find_hits(otus)
#     seqs = seqs.values()
#     out = set()
#     for overlap in overlaps:
#         all_genome_seqs = table[table['genome'] == overlap]['16s_sequence']
#         hits = [x in seqs for x in all_genome_seqs]
#         if np.all(hits):
#             out.append(overlap)
#     return out

# def find_hits(otus, table=t):
#     all = set()
#     overlaps = set()
#     seqs = {}
#     for otu in otus.index:
#         seq = t[t['feature'] == otu]['16s_sequence'].values[0]
#         matches = set(t[t['16s_sequence'] == seq]['genome'])
#         if len(matches & all) > 0:
#             overlaps = overlaps | (matches & all)
#         all = all | matches
#         seqs[otu] = seq

#     return overlaps, seqs


def solve_sequences(candidates):
    raise NotImplementedError()

def solve_all(path, db_path, true_values=None):
    db = pd.read_pickle(db_path)
    table = load_table(path)

    out = pd.DataFrame(columns=["sample", "genome", "ptr"])
    for column in table.columns:
        sample = table[column]
        candidates = find_candidates(table, db)
        solutions = solve_sequences(candidates)
        # TODO: append outputs
        for genome, ptr in solutions:
            out = out.append({"sample" : column, "genome" : genome, "ptr" : ptr}, ignore_index=True)
        # TODO: append true values

    return out