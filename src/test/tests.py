"""
Various tests for 16S utils
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ..util.simulation import *
from ..db import *
import timeit

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