"""
Compute PTR estimates for the "complete_1e*" directories
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.solver import solver

my_db = pd.read_pickle("./data/db.pkl")
my_collisions = pd.read_pickle("./data/collisions.pkl")
my_otus = pd.read_table("./out/6bc6c418-9e20-4b2b-93a5-603a912a128d/16s_otus.tsv", dtype={0: str})
my_otus = my_otus.set_index('otu')

my_true_ptrs = pd.read_table("./out/6bc6c418-9e20-4b2b-93a5-603a912a128d/ptrs.tsv", dtype={0: str})
my_true_ptrs = my_true_ptrs.set_index('genome')

def md5_to_genomes(
    md5 : str,
    database : pd.DataFrame,
    collisions : pd.DataFrame) -> (list, list):
    """
    TODO: Remove this (it is now a method of the RnaDatabase class)

    Given an OTU's 16S md5 hash, return all genomes in which it is contained

    Args:
    -----
    md5:
        String. The md5 hash of a given 16S sequence.
    database:
        Pandas DataFrame. A matrix of 16S positions and sequences.
    collisions:
        Pandas DataFrame. A matrix of conflicting 16S sequences (spoiler genes).

    Returns:
    --------
    The following two lists:
    * db_matches: a list of genome IDs matching this OTU in 'database'
    * db_collisions: a list of genome IDs matching this OTU in 'collisions'

    Raises:
    -------
    TODO
    """

    # Check DB
    db_matches = database[database['16s_md5'] == md5]['genome'].unique()

    # Check collisions
    db_collisions = collisions[collisions['16s_md5'] == md5]['genome'].unique()

    return list(db_matches), list(db_collisions)

def solve_genome(
    genome_id : str,
    sample_id : str,
    database : pd.DataFrame,
    otus : pd.DataFrame,
    true_ptrs : pd.DataFrame) -> dict:
    """
    Given a genome ID, a DB, and some coverages, estimate the PTR.

    Args:
    -----
    genome_id:
        String. Genome ID as provided by md5_to_genomes().
    sample_id:
        Integer or string. Sample number for the given genome.
    database:
        Pandas DataFrame. A matrix of 16S positions and sequences.
    otus:
        Pandas DataFrame. A matrix of 16S OTU read/abundance counts.
    true_ptrs:
        Pandas DataFrame. A matrix of true PTR values, if known.

    Returns:
    --------
    A dict with keys 'genome', 'sample', 'ptr', 'true_ptr' to be appended to a results DB.

    Raises:
    -------
    TODO
    """
    # Build up x_values
    db_matched = database[database['genome'] == genome_id]
    x_positions = db_matched['16s_position'] / db_matched['size']

    # Build up mappings
    md5s_matched = db_matched['16s_md5']
    mapping = {}
    idx = 0
    # TODO: vectorize this
    for md5 in md5s_matched:
        if md5 not in mapping:
            mapping[md5] = idx
            idx += 1
    x_mapping = [mapping[x] for x in md5s_matched]

    # Sort md5s by their index
    md5s = [None for _ in mapping]
    for md5 in mapping:
        md5s[mapping[md5]] = md5

    # Build up coverages
    coverages = otus[sample_id].reindex(md5s)

    # Send to solver
    results = solver(x_positions, x_mapping, coverages)

    # Append output to PTRs dataframe
    m = results[0]
    b = results[1]
    peak = np.exp2(b)
    trough = np.exp2(m * 0.5 + b)
    ptr = peak / trough

    # Get true PTR
    try:
        true_ptr = true_ptrs.loc[genome_id, sample_id]
    except KeyError as e:
        true_ptr = np.nan
        print(e)

    return {"genome" : genome_id, "sample" : sample_id, "ptr" : ptr, "true_ptr" : true_ptr}

def solve_sample(
    sample_id : str,
    database : pd.DataFrame,
    otus : pd.DataFrame,
    collisions : pd.DataFrame,
    true_ptrs : pd.DataFrame) -> list:
    """
    Given a sample name, solve all available 16S systems.

    Args:
    -----
    sample_id:
        String. The name of a given sample. Used for indexing into OTU matrix.
    database:
        Pandas DataFrame. A matrix of 16S positions and sequences.
    otus:
        Pandas DataFrame. A matrix of 16S OTU read/abundance counts.
    collisions:
        Pandas DataFrame. A matrix of conflicting 16S sequences (spoiler genes).
    true_ptrs:
        Pandas DataFrame. A matrix of true PTR values, if known.

    Returns:
    --------
    A list of dicts (solve_genome outputs) to be appended to an output DB.

    Raises:
    -------
    TODO
    """
    genomes = []
    out = []
    # Build up lists of genomes and md5s
    for md5 in otus.index:
        if otus.loc[md5, sample_id] > 0:
            match, coll = md5_to_genomes(md5, database=database, collisions=collisions)

            # Skip collisions for now... these are overdetermined
            if coll:
                print(f"Collision for {md5}. Skipping this sequence...")
            else:
                genomes += match

    for genome_id in set(genomes):
        result = solve_genome(genome_id, sample_id, database=database, otus=otus, true_ptrs=true_ptrs)
        out.append(result)

    return out

def solve_matrix(
    database : pd.DataFrame,
    otus : pd.DataFrame,
    collisions : pd.DataFrame,
    true_ptrs : pd.DataFrame = None) -> pd.DataFrame:
    """
    Given a 16S db, OTU read/abundance matrix, and true PTR values (optional), esimate PTRs.

    Args:
    -----
    database:
        Pandas DataFrame. A matrix of 16S positions and sequences.
    otus:
        Pandas DataFrame. A matrix of 16S OTU read/abundance counts.
    collisions:
        Pandas DataFrame. A matrix of conflicting 16S sequences (spoiler genes).
    true_ptrs:
        Pandas DataFrame. A matrix of true PTR values, if known.

    Returns:
    --------
    A Pandas Dataframe with the following columns:
    * genome: genome ID (taken from database)
    * sample: sample ID (as given in OTU matrix)
    * ptr: PTR (estimated by tool)
    * true_ptr: PTR (provided in true_ptrs DataFrame)
    * err: absolute error (capped at 5) between true and estimated PTR

    Raises:
    -------
    TODO
    """
    out = pd.DataFrame(columns=["genome", "sample", "ptr", "true_ptr"])

    # For each column, build up x_values, mappings, coverages; send to solver
    for column in otus.columns:
        results = solve_sample(column, database=database, otus=otus, collisions=collisions, true_ptrs=true_ptrs)
        out = out.append(results, ignore_index=True)

    out['err'] = np.abs(out['ptr'] - out['true_ptr'])
    # Cut off error threshold
    out['err'] = out['err'].apply(lambda x: np.min([x,5]))

    return out


# Solve PTRs as provided in 'otus.tsv':
ptrs = solve_matrix(my_db, my_otus, my_collisions, my_true_ptrs)
plt.matshow(ptrs.pivot('genome', 'sample', 'err'))
plt.colorbar()
plt.show()
