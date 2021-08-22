"""
Scripts to simulate read coverage from a known PTR
"""

import gzip
from uuid import uuid4
import os
from pprint import pprint

import numpy as np
import pandas as pd

from Bio import SeqIO

def ptr_curve(
    size : int,
    ptr : float,
    oor : int = 0) -> np.ndarray:
    """
    Given an array of x-values and a PTR, produce a plausible PTR curve.
    This function assumes that origin of replication is at x=0.

    The function for PTR coverage curve is:
        p(x) = ptr^(-2x + 1) / A
        A = sum(ptr^(-2x + 1)) over x (normalization constant)

    Args:
    -----
    size:
        Integer. How many positions to generate PTR curve for. Should correspond to genome size.
    ptr:
        Float. Peak-to-trough ratio of coverage curve. Should be at least 1.
    oor:
        Int. Index at which coverage peak/origin of replication is found. Should be between 0 and size.

    Returns:
    --------
    x_array:
        A numpy array of x-coordinate positions.
    y_array:
        A numpy array of read probabilities corresponding to each position in x_array.

    Raises:
    -------
    TODO
    """

    # Check size is positive
    if size < 1:
        raise Exception("Size must be a positive integer")
    # Check 0 < OOR < size
    if oor > size or oor < 0:
        raise Exception("OOR must be in range 0 < OOR < size")

    # Initialize array in [0, 1) interval
    x_array = np.linspace(0, 1, size)
    x_original = x_array.copy()

    # Reflect about trough for values not in the first half
    x_array[np.where(x_array > 0.5)] = 1 - x_array[np.where(x_array > 0.5)]

    # Return the normalized probability. Here the coverage at the peak is the PTR, and the coverage at the trough is 1.
    y_array = np.power(ptr, -2 * x_array) * ptr

    # Normalize array
    y_array = y_array / np.sum(y_array)

    # Return array, adjusting for OOR position
    return x_original, np.append(y_array[oor:], y_array[:oor])

def rc(seq):
    """
    Returns the reverse complement of a sequence.

    Args:
    -----
    seq:
        A string corresponding to a nucleotide sequence.

    Returns:
    --------
    The reverse-complement of a string, in caps.

    Raises:
    -------
    TODO
    """
    seq = seq.lower()
    seq = seq.replace("a", "T")
    seq = seq.replace("c", "G")
    seq = seq.replace("g", "C")
    seq = seq.replace("t", "A")
    return seq[::-1]

def generate_reads(
    sequence : str,
    n_reads : int,
    db : pd.DataFrame = None,
    read_length : int = 300,
    ptr : float = 1,
    oor : int = 0,
    name : str = "") -> list:
    """
    Generates synthetic reads from a given sequence.

    Args:
    -----
    db:
        Pandas dataframe containing 16S information. SHOULD BE FILTERED!
    sequence:
        String (or Biopython Seq object) corresponding to the full nucleotide sequence of an organism/contig.
    n_reads:
        Integer. How many reads to draw from this sequence.
    read_length:
        Integer. How many base-pairs to draw per read.
    ptr:
        Float. Peak-to-trough ratio of the organism.
    oor:
        Integer. At which position to simulate the coverage peak.
    name:
        String. What name to give this organism in the simulated reads.

    Returns:
    --------
    A list of simulated fastq reads. Each read is a single string with the following format:
        @{name}:{index}:{start}:{end}
        ACTGACTG...
        +
        IIIIIIII...

    Raises:
    -------
    TODO
    """

    # Ensure OOR is int
    oor = int(oor)

    # Get 16S positions from DB
    if db is not None:
        rnas = list(db['16s_position'])
        rna_reads = {x : 0 for x in db['16s_md5'].unique()}
    else:
        print(f"No RNAs found for genome: {name}")
        rnas = []
        rna_reads = {}

    # Account for circularity of chromosome
    seq_length = len(sequence)
    seq_repeat = sequence[0:read_length]
    new_seq = sequence + seq_repeat
    new_seq = new_seq.lower() # just to distinguish between rc and forward strand

    # Sample starts from the ptr-adjusted distribution
    x, probs = ptr_curve(seq_length, ptr, oor)
    positions = range(seq_length)

    starts = np.random.choice(positions, p=probs, size=n_reads)

    # Given starts, make sequences
    reads = []
    for idx, start in enumerate(starts):
        # Get the read
        end = start + read_length
        read = str(new_seq[start:end])

        # Add reverse complement --- patch #2 04.07.2021
        if np.random.rand() > .5:
            read = rc(read)

            # Need to change indexing
            old_start = start
            start = end
            end = old_start

        # Check for RNA membership --- patch #3 08.22.2021
        dists = [np.abs(rna - start) for rna in rnas]
        if dists and np.min(dists) < read_length:
            # TODO: redo this using strand, start, and stop
            otu_name = db.iloc[np.argmin(dists)]['16s_md5']
            rna_reads[otu_name] += 1

        # Concatenate into a plausible-looking fastq output and push to output
        fastq_line1 = f"@{name}:{idx}:{start}:{end}"
        fastq_line2 = read
        fastq_line3 = '+'
        fastq_line4 = read_length * 'I' # Max quality, I guess?
        reads.append("\n".join([fastq_line1, fastq_line2, fastq_line3, fastq_line4]))

    return reads, rna_reads

def simulate(
    db : pd.DataFrame,
    sequences : dict,
    ptrs : np.array = None,
    coverages : np.array = None,
    n_samples : int = 10,
    read_length : int = 300,
    scale : float = 1e5,
    verbose : bool = True) -> (list, np.array, np.array):
    """
    Given known PTRs and coverages, generate synthetic reads.

    TODO: Adapt to multiple contigs

    Args:
    -----
    TODO

    Returns:
    --------
    TODO

    Raises:
    -------
    TODO
    """

    rng = np.random.default_rng() #random number generator to shuffle
    inputs = pd.DataFrame(columns=["Species", "Sample", "PTR", "Reads"])

    samples = []
    otu_matrix = []

    # Randomly choose PTRs
    if ptrs is None:
        ptrs = 1 + np.random.rand(len(sequences), n_samples)

    # Randomly choose coverages
    if coverages is None:
        coverages = np.random.exponential(scale=scale, size=(len(sequences), n_samples))
        coverages = coverages.astype(int)

    for sample_no in range(n_samples):
        sample = []

        for idx, genome in enumerate(sequences):
            ptr = ptrs[idx, sample_no]
            n_reads = coverages[idx, sample_no]

            inputs = inputs.append({
                "Sample" : sample_no,
                "Species" : genome,
                "PTR" : ptr,
                "Reads" : n_reads
            }, ignore_index=True)

            try:
                start = db[db["genome"] == genome]['oor_position'].iloc[0]
            except KeyError:
                print(f"No OOR found for {genome}, assume OOR at 0.")
                start = 0

            if np.isnan(start):
                print(f"No OOR found for {genome}, assume OOR at 0.")
                start = 0

            if verbose:
                print(f"Generating sample {sample_no} for organism {genome}...")

            seq = sequences[genome]

            # This aggregates all reads into a single list to be shuffled later
            reads, rna_reads = generate_reads(
                sequence=seq,
                n_reads=n_reads,
                db=db[db['genome'] == genome],
                ptr=ptr,
                name=genome,
                oor=start,
                read_length=read_length
            )
            sample += reads
            otu_matrix.append(rna_reads)

        rng.shuffle(sample)
        samples.append(sample)

    return samples, ptrs, coverages, otu_matrix

def write_output(
    samples : list,
    ptrs : np.array = None,
    coverages : np.array = None,
    path : str = None,
    prefix : str = 'S_',
    use_gzip : bool = True) -> None:
    """
    Write a set of reads as a fastq.gz file

    Args:
    -----
    TODO

    Returns:
    --------
    None (writes to disk)

    Raises:
    TODO
    """

    # Set path by UUID if needed
    if path is None:
        path = f"./out/{uuid4()}/"
        os.mkdir(path)

    # Save PTRs if given
    if ptrs is not None:
        np.savetxt(f"{path}/ptrs.tsv", ptrs, delimiter="\t")
        print(f"Finished writing PTRs to {path}/ptrs.tsv")

    # Save coverages if given
    if coverages is not None:
        np.savetxt(f"{path}/coverages.tsv", coverages, delimiter="\t")
        print(f"Finished writing coverages to {path}/coverages.tsv")

    # Save reads
    if samples is not None:
        for idx, sample in enumerate(samples):
            if use_gzip:
                with gzip.open(f"{path}{prefix}{idx}.fastq.gz", "wb") as f:
                    f.write("\n".join(sample).encode())
            else:
                with open(f"{path}{prefix}{idx}.fastq", "wb") as f:
                    f.write("\n".join(sample).encode())

            print(f"Finished writing sample {idx} to {path}{prefix}{idx}.fastq.gz")


def simulate_from_ids(
    db : pd.DataFrame,
    ids : list,
    fasta_path : str,
    suffix : str = '.fna.gz',
    **simulate_args) -> (list, np.ndarray, np.ndarray):
    """
    Given a list of IDs, simulate reads.

    Args:
    -----
    TODO

    Returns:
    --------
    TODO

    Raises:
    -------
    TODO
    """

    # Create dict of sequences
    sequences = {}

    for gid in ids:
        path = f"{fasta_path}/{gid}{suffix}"
        sequences[gid] = []

        if suffix.endswith('gz'):
            with gzip.open(path, "rt") as handle:
                sequence = SeqIO.parse(handle, "fasta")
                for record in sequence.records:
                    sequences[gid].append(record.seq)

        else:
            sequence = SeqIO.parse(path, "fasta")
            for record in sequence.records:
                sequences[gid].append(record.seq)

    # TODO: remove reliance on this
    sequences = {seq : sequences[seq][0] for seq in sequences}
    # This only works for complete genomes

    return simulate(db=db, sequences=sequences, **simulate_args)
