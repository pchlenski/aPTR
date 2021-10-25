"""
Various tests for 16S utils
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ..util.simulation import *
from ..db import *
import timeit
from src.multi_solver import multi_solver

def test_1():
    """
    Draw a plot of a dramatic PTR curve.

    Should look like this:
        /-\
     \-/
    """
    x, y = ptr_curve(22, 2, 6)
    print("Sum of y:", np.sum(y))
    plt.plot(x, y)
    plt.show()

def test_2():
    """
    Simulate two reads. Tests generate_reads().
    """
    db = pd.read_pickle('./data/db.pkl')
    seq = 'actgactgactgactgactgactgactgactgactgactgactgactg'
    out = generate_reads(
        db=db,
        sequence=seq, 
        n_reads=2,
        read_length=10,
        ptr=2,
        name="test")
    print(out)

def test_3(write=False):
    """
    Simulate two samples. Tests simulate() and write_fastq().
    """

    # Fake sequences
    seq1 = 'actgactgactgactgactgactgactgactgactgactgactgactg'
    seq2 = rc(seq1)
    seqs = {'seq1':seq1, 'seq2':seq2}

    # Fake database --- we only need names and OORs
    db = pd.DataFrame()
    db = db.append([
        {'genome':'seq1', 'oor_position':0},
        {'genome':'seq2', 'oor_position':10}
    ], ignore_index=True)

    # Force 2 reads per column
    coverages = np.array([[2,2],[2,2]])

    reads, ptrs, coverages = simulate(
        db=db,
        sequences=seqs,
        n_samples=2,
        coverages=coverages,
        read_length=5
    )

    # Print outputs
    for idx, x in enumerate(reads):
        for y in x:
            print(y)
        print("\n")

    print(np.array(reads).shape) # Should be 2,4
    print(ptrs.shape) # Should be 2,2
    print(coverages.shape) # Should be 2,2

    # Save fastqs
    write_output(reads, ptrs=ptrs, coverages=coverages)
    write_output(reads, ptrs=ptrs, coverages=coverages, use_gzip=False) # Test both ways

def test_4():
    """
    Simulate two samples from real fasta files. Tests simulate_from_ids().

    This test uses single-contig sequences.
    """
    db = pd.read_pickle('./data/db.pkl')
    simulate_from_ids(
        db=db,
        ids=['703.8', '732.8'],
        fasta_path='/Users/phil/Documents/Columbia/16s-ptr/data/seqs',
        suffix='.fna.gz',
        n_samples=2,
    )

def test_5():
    """
    Simulate two samples from real fasta files. Tests simulate_from_ids().

    This test uses multiple-contig draft genomes.
    """
    db = pd.read_pickle('./data/db.pkl')
    simulate_from_ids(
        db=db,
        ids=['192.7', '562.7382'],
        fasta_path='/Users/phil/Documents/Columbia/16s-ptr/data/seqs',
        suffix='.fna.gz',
        n_samples=2,
    )

def test_6(dd="./data/", ex_dir="./out/complete/", scale=1):
    """
    Generate an OTU matrix
    """

    # db = pd.read_pickle('./data/db.pkl')
    # print(db.columns)

    db = RnaDB(f"{dd}db.pkl", f"{dd}collisions.pkl")

    ptrs = pd.read_table(f"{ex_dir}ptrs.tsv", dtype={0: str})
    ptrs = ptrs.set_index("genome")

    covs = pd.read_table(f"{ex_dir}coverages.tsv", dtype={0: str})
    covs = covs.set_index("genome")

    matrix = generate_otu_matrix(db, ptrs, covs, scale=scale)
    print(matrix)

def test_7(database, genome='325240.15'):
    """
    Generate and test coverage from a genome
    """

    genome_rows = database[genome]
    x_vals = genome_rows['16s_position'] / genome_rows['size']
    size = genome_rows['size'].iloc[0]

    ptr = 1 + np.random.rand()
    ptrs = pd.DataFrame(columns=["sample1"])
    ptrs.loc[genome,"sample1"] = ptr

    coverage = 1000000
    coverages = pd.DataFrame(columns=["sample1"])
    coverages.loc[genome,"sample1"] = coverage
    otus = generate_otu_matrix(database, ptrs, coverages)

    # normalize
    otus["sample1"] = otus["sample1"] / np.sum(otus["sample1"])

    ptr_est = database.solve_matrix(otus)["ptr"].iloc[0]

    print(f"True PTR: {ptr}, Estimated PTR: {ptr_est}")

def test_8(database, genomes=['325240.15', '407976.7']):#, '407976.7', '693973.6']):
    """
    Generate and test coverage from two entangled genomes, individually then together

    Good set to use: ['325240.15', '407976.7', '407976.7', '693973.6']
    """

    ptrs = pd.DataFrame(columns=["sample1"])
    coverages = pd.DataFrame(columns=["sample1"])

    for genome in genomes:
        ptrs.loc[genome,"sample1"] = np.random.rand()
        coverages.loc[genome,"sample1"] = 100000

    otus = generate_otu_matrix(database, ptrs, coverages)

    # build up x_values_list, mappings_list
    dbg2 = database[genomes]
    # md5s = dbg['16s_md5']
    md5s = list(otus.index)
    mapping = {}
    idx = 0
    for md5 in md5s:
        if md5 not in mapping:
            mapping[md5] = idx
            idx += 1
            # TODO: VECTORIZE

    mappings_list = [] 
    x_values_list = []
    for genome in genomes:
        # get x values
        dbg = database[genome]
        x_values = dbg['16s_position'] / dbg['size']
        x_values = list(x_values)
        x_values_list.append(x_values)

        # get md5
        x_map = [mapping[x] for x in dbg['16s_md5']]
        mappings_list.append(x_map)

    # get coverages
    coverages = list(otus['sample1'])

    print("X VALUES", x_values_list)

    solution = multi_solver(x_values_list, mappings_list, coverages)
    print(solution)




