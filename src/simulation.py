""" Scripts for generating synthetic data with known PTRs and abundances """

import os
import gzip
from collections import Counter
from typing import Tuple, List, Dict
import pandas as pd
import numpy as np
from Bio import SeqIO
from uuid import uuid4
from .database import RnaDB


def ptr_curve(
    size: int, ptr: float, oor: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given an array of x-values and a PTR, produce a plausible PTR curve.
    This function assumes that origin of replication is at x=0.

    The function for PTR coverage curve is:
        p(x) = ptr^(-2x + 1) / A
        A = sum(ptr^(-2x + 1)) over x (normalization constant)

    Args:
    -----
    size:
        Integer. How many positions to generate PTR curve for. Should correspond
        to genome size.
    ptr:
        Float. Peak-to-trough ratio of coverage curve. Should be at least 1.
    oor:
        Int. Index at which coverage peak/origin of replication is found. Should
        be between 0 and size.

    Returns:
    --------
    x_array:
        A numpy array of x-coordinate positions.
    y_array:
        A numpy array of read probabilities corresponding to each position in
        x_array.

    Raises:
    -------
    TODO
    """

    # Check size is positive
    if size < 1:
        raise Exception("Size must be a positive integer")

    # Check 0 < OOR < size
    if oor > size or oor < 0:
        raise Exception(
            f"OOR must be in range 0 < OOR < size. Was given OOR={oor} and size={size}"
        )

    # Fix NaN:
    if np.isnan(oor):
        oor = 0

    # Coerce OOR and size to int
    size = int(size)
    oor = int(oor)

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


def reverse_complement(seq):
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
    sequence: str,
    n_reads: int,
    db: pd.DataFrame = None,
    read_length: int = 300,
    ptr: float = 1.0,
    oor: int = 0,
    name: str = "",
    fastq: bool = False,
) -> list:
    """
    Generates synthetic reads from a given sequence.

    Args:
    -----
    db:
        Pandas dataframe containing 16S information. SHOULD BE FILTERED!
    sequence:
        String (or Biopython Seq object) corresponding to the full nucleotide
        sequence of an organism/contig.
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
    fastq:
        Boolean. Whether to generate fastq reads.

    Returns:
    --------
    A list of simulated fastq reads. Each read is a single string with the
    following format:
        @{name}:{index}:{start}:{end}
        ACTGACTG...
        +
        IIIIIIII...

    Raises:
    -------
    TODO
    """

    # Use RNG for speed improvements
    rng = np.random.default_rng()

    # Ensure OOR is int
    oor = int(oor)

    # Get 16S positions from DB
    if db is not None:
        rnas = list(db["16s_position"])
        rna_reads = {x: 0 for x in db["md5"].unique()}
    else:
        print(f"No RNAs found for genome: {name}")
        rnas = []
        rna_reads = {}

    # Account for circularity of chromosome
    seq_length = len(sequence)
    seq_repeat = sequence[0:read_length]
    new_seq = sequence + seq_repeat
    new_seq = (
        new_seq.lower()
    )  # just to distinguish between rc and forward strand

    # Sample starts from the ptr-adjusted distribution
    _, probs = ptr_curve(seq_length, ptr, oor)
    positions = np.arange(seq_length)

    # starts = np.random.choice(positions, p=probs, size=n_reads)
    starts = rng.choice(positions, p=probs, size=n_reads)

    # Vectorizing saves a lot of time, e.g. we can do 1e6 reads in 20s vs 2m
    ends = starts + read_length

    # This part can't fully be vectorized
    if fastq:
        # Get reads
        reads = [str(new_seq[start:end]) for start, end in zip(starts, ends)]

        # Add reverse complement for 50% of reads
        rev_mask = np.random.rand(n_reads) > 0.5
        reads = np.array(reads)
        reads[rev_mask] = [reverse_complement(read) for read in reads[rev_mask]]

    # Check for RNA membership
    # TODO: double-check this logic, stepping through it with a debugger
    dists = np.abs(
        np.array(rnas, dtype=int) - starts[:, None]
    )  # vectorized to 2D; dtype=int to avoid OOM errors
    rna_mask = np.min(dists, axis=1) < read_length
    rna_names = db.iloc[np.argmin(dists, axis=1)]["md5"]
    for name in rna_names[rna_mask]:
        rna_reads[name] += 1

    # Concatenate into a plausible-looking fastq output and push to output
    if fastq:
        reads_out = []
        for idx, (start, end, read) in enumerate(zip(starts, ends, reads)):
            fastq_line1 = f"@{name}:{idx}:{start}:{end}"
            fastq_line2 = read
            fastq_line3 = "+"
            fastq_line4 = read_length * "I"
            reads_out.append(
                "\n".join([fastq_line1, fastq_line2, fastq_line3, fastq_line4])
            )
    else:
        reads_out = None

    return reads_out, rna_reads


def simulate(
    db: pd.DataFrame,
    sequences: Dict[str, List[str]],
    ptrs: np.array = None,
    coverages: np.array = None,
    n_samples: int = 10,
    read_length: int = 300,
    scale: float = 1e5,
    verbose: bool = True,
    shuffle: bool = True,
    fastq: bool = True,
) -> Tuple[List[List[str]], np.array, np.array, np.array]:
    """
    Given known PTRs and coverages, generate synthetic reads.
    TODO: Adapt to multiple contigs

    Args:
    -----
    db:
        A dataframe containing per-contig 16S sequence and position information
    sequences: dict
        A dict mapping genome IDs to lists of contig sequences.
    ptrs: np.array
        A #{genomes} x #{samples} array of true PTRs.
    coverages: np.array
        A #{genomes} x #{samples} array of read counts.
    n_samples: int
        How many samples to generate.
    read_length: int
        How many base pairs to make each read.
    scale: float
        Scaling factor for exponential distribution during read sampling.
    verbose: bool
        Report on data generation process.
    shuffle: bool
        If true, will shuffle the order of genomes in the output. Suppress this
        to prevent OOM errors when generating large datasets.
    fastq: bool
        If true, will return reads and an OTU matrix.

    Returns:
    --------
    samples:
        A list of samples. Each sample is a list of strings, each of which is a
        single read.
    ptrs:
        Numpy array. A #{genomes} x #{samples} array of true PTRs. If the 'ptrs'
        argument is set, returns that.
    coverages:
        Numpy array. A #{genomes} x #{samples} array of read counts. If the
        'coveragees' argument is set, returns that.
    otu_matrix:
        Numpy array. A #{OTUs} x #{samples} array of read counts, downsampled
        from all reads.

    Raises:
    -------
    TODO
    """

    rng = np.random.default_rng()  # random number generator to shuffle
    inputs = pd.DataFrame(columns=["Species", "Sample", "PTR", "Reads"])

    samples = []
    otu_matrix = pd.DataFrame(columns=range(n_samples))

    # Randomly choose PTRs
    if ptrs is None:
        ptrs = 1 + np.random.rand(len(sequences), n_samples)

    # Randomly choose coverages
    if coverages is None:
        coverages = np.random.exponential(
            scale=scale, size=(len(sequences), n_samples)
        )
        coverages = coverages.astype(int)

    for sample_no in range(n_samples):
        sample = []
        sample_rna = []

        for idx, genome in enumerate(sequences):
            ptr = ptrs[idx, sample_no]
            n_reads = coverages[idx, sample_no]

            inputs = inputs.append(
                {
                    "Sample": sample_no,
                    "Species": genome,
                    "PTR": ptr,
                    "Reads": n_reads,
                },
                ignore_index=True,
            )

            try:
                start = db[db["genome"] == genome]["oor_position"].iloc[0]
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
            # TODO: Add the option of calling generate_reads_contig instead
            reads, rna_reads = generate_reads(
                sequence=seq,
                n_reads=n_reads,
                db=db[db["genome"] == genome],
                ptr=ptr,
                name=genome,
                oor=start,
                read_length=read_length,
                fastq=fastq,
            )
            if fastq:
                sample += reads

            sample_rna.append(rna_reads)

        print("Sample RNA:")
        print(sample_rna)
        otu_matrix[sample_no] = pd.DataFrame(sample_rna).sum(axis=0)

        if shuffle:
            rng.shuffle(sample)

        if fastq:
            samples.append(sample)
        else:
            samples = None

        # Better output for OTU matrix
        otu_matrix = pd.DataFrame(otu_matrix)  # , dtype=int).T

    return samples, ptrs, coverages, otu_matrix


def simulate_from_ids(
    db: pd.DataFrame,
    ids: list,
    fasta_path: str,
    suffix: str = ".fna.gz",
    fastq: bool = True,
    **simulate_args,
) -> Tuple[list, np.ndarray, np.ndarray, list]:
    """
    Given a list of IDs, simulate reads.

    Args:
    -----
    db:
        A dataframe containing per-contig 16S sequence and position information
    ids:
        A list of genome IDs for which to generate reads.
    fasta_path:
        String. The path to the directory containing fasta files.
    suffix:
        Sring. The suffix for fasta files.
    fastq: bool
        If True, geneerates actual fastq reads. Otherwise uses None
    **simulate_args:
        Arguments for the simulate() function

    Returns:
    --------
    Same outputs as simulate()

    Raises:
    -------
    TODO
    """

    # Create dict of sequences
    sequences = {}

    if fastq:
        for gid in ids:
            path = f"{fasta_path}/{gid}{suffix}"
            sequences[gid] = []

            if suffix.endswith("gz"):
                with gzip.open(path, "rt") as handle:
                    sequence = SeqIO.parse(handle, "fasta")
                    for record in sequence.records:
                        sequences[gid].append(record.seq)

            else:
                sequence = SeqIO.parse(path, "fasta")
                for record in sequence.records:
                    sequences[gid].append(record.seq)

        # TODO: remove reliance on this
        sequences = {seq: sequences[seq][0] for seq in sequences}
        # This only works for complete genomes

    # A workaround to remove dependence on fasta files when fastq is False
    else:
        genome_lengths = (
            db[db["genome"].isin(ids)].groupby("genome").max()["size"]
        )  # Using .max() is hacky, but they should all be the same value

        sequences = {gid: "N" * genome_lengths[gid] for gid in ids}
        # Need a fake value

    return simulate(db=db, sequences=sequences, fastq=fastq, **simulate_args)


def generate_reads_contig(
    sequences: list,
    n_reads: int,
    db: pd.DataFrame = None,
    read_length: int = 300,
    ptr: float = 1,
    oor: int = 0,
    name: str = "",
) -> list:
    """
    TODO: Develop code for simulating reads from contig. Note: we can't assume we know OOR
    Should it use OOR and genome length where possible?
    """
    raise NotImplementedError


def generate_otu_matrix(
    db: RnaDB,
    ptrs: pd.DataFrame,
    coverages: pd.DataFrame,
    scale: float = 1,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Given coverages and PTRs, generate an OTU matrix with custom scaled coverage.

    Args:
    -----
    db:
        An RnaDB object.
    ptrs:
        Numpy array. A #{genomes} x #{samples} array of true PTRs.
    coverages:
        Numpy array. A #{genomes} x #{samples} array of read counts.
    scale:
        Float. Multiplier for read counts in coverages matrix.
    verbose:
        Boolean. Prints progress updates if true.

    Returns:
    --------
    A #{genomes} x #{samples} dataframe of 16S read counts.

    Raises:
    -------
    TODO
    """
    n_rows, n_cols = ptrs.shape

    if ptrs.shape != coverages.shape:
        raise Exception("Coverage and PTR shapes do not match!")

    out = []

    for row_idx in range(n_rows):
        if verbose == True:
            print(f"Row: {row_idx}")

        gid_ptr = ptrs.index[row_idx]
        gid_cov = coverages.index[row_idx]

        # Check that genome IDs match
        if gid_ptr != gid_cov:
            raise Exception(f"ID mismatch: '{gid_ptr}' != '{gid_cov}'")

        genome_db = db[gid_ptr]
        size = genome_db.iloc[0]["size"]
        oor = genome_db.iloc[0]["oor_position"]

        # Check coverage
        for col_idx in range(n_cols):
            # if "genome" not in [ptrs.columns[col_idx], coverages.columns[col_idx]]:
            ptr = ptrs.iloc[row_idx, col_idx]
            coverage = coverages.iloc[row_idx, col_idx]

            if coverage > 0:
                starts = list(genome_db["16s_position"])
                _x, curve = ptr_curve(size, ptr, oor)

                # Downsample curve and renormalize
                probs = curve[starts]
                probs /= np.sum(probs)

                # Sample
                md5s = list(genome_db["md5"])
                sample = np.random.choice(
                    md5s, size=int(coverage * scale), p=probs
                )

                # Output to dataframe
                counter = Counter(sample)
                for key in counter:
                    out.append(
                        {
                            "otu": key,
                            "sample": col_idx,
                            "count": int(counter[key]),
                        }
                    )

    # pivot = pd.DataFrame(out).pivot("otu", "sample", "count")
    pivot = pd.pivot_table(
        data=pd.DataFrame(out),
        index="otu",
        columns="sample",
        values="count",
        aggfunc=np.sum,
    )
    pivot.columns = ptrs.columns

    return pivot


def write_output(
    samples: list,
    ptrs: np.array = None,
    coverages: np.array = None,
    path: str = None,
    prefix: str = "S_",
    use_gzip: bool = True,
) -> None:
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

            print(
                f"Finished writing sample {idx} to {path}{prefix}{idx}.fastq.gz"
            )
