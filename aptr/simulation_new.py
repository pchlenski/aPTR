"""There were problems with the old simulation code, so I wrote some more concise stuff instead."""

import numpy as np
import pandas as pd
from Bio import SeqIO
from typing import Iterable, List, Tuple
import gzip
from aptr.database import RnaDB
from aptr.oor_distance import oor_distance
from aptr.string_operations import rc


def _exact_coverage_curve(log_ptr, distances=None, oor=None, locations=None, size=1):
    # Coercion:
    if distances is not None:
        distances = np.array(distances)
    if locations is not None:
        locations = np.array(locations)
    if log_ptr is not None:
        log_ptr = np.array(log_ptr)

    if distances is not None and distances.ndim == 1:
        distances = distances[:, None]
    if locations is not None and locations.ndim == 1:
        locations = locations[:, None]
    if oor is not None and isinstance(oor, (int, float)):
        oor = np.array([oor])
    if size is not None and isinstance(size, (int, float)):
        size = np.array([size])

    # Respect constraints on distances:
    if distances is None:
        # Can infer distances from OOR and locations
        if locations is not None and oor is not None:
            if (np.max(locations) > size).any() or (oor > size).any():
                raise ValueError("Size must be larger than locations and oor")
            if (np.min(locations) < 0).any() or (oor < 0).any():
                raise ValueError("Location and oor can not be < 0")
            oor_distances = oor_distance(locations, oor, size)
        else:
            raise ValueError("Must provide locations, size, and OOR")

    # Enforce normalization
    elif np.max(distances) > 1 or np.min(distances) < 0:
        raise ValueError("OOR distances must be normalized.")

    else:
        oor_distances = np.array(distances)

    return np.exp(-log_ptr * oor_distances)  # Half the size, uses abundance more reasonably


def _exact_coverage_curve_genome(genome, log_ptr, db=None):
    """Given a genome ID and a log-PTR, generate coverages at 16S positions"""
    db = RnaDB() if db is None else db

    # Input validation
    genome = str(genome)
    log_ptr = np.array(log_ptr)
    if not isinstance(db, RnaDB):
        raise TypeError("db must be an RnaDB")

    # Use DB to get locations and return coverages
    size = db[genome]["size"].max()
    rna_locations = np.array(db[genome]["16s_position"] / size)
    oor = float(db[genome]["oor_position"].max() / size)

    return rna_locations, _exact_coverage_curve(log_ptr, oor=oor, locations=rna_locations)


def _coverage_16s_and_wgs(genome, log_ptr, db=None):
    """Given a genome ID and a log-PTR, simulate 16S and WGS coverage simultaneously"""
    db = RnaDB() if db is None else db

    # Input validation
    genome = str(genome)
    log_ptr = np.array(log_ptr)
    if not isinstance(db, RnaDB):
        raise TypeError("db must be an RnaDB")

    # Find number of wgs reads from get_genome_reads:
    size = db[genome]["size"].max()
    oor = db[genome]["oor_position"].max() / size
    rna_locations, rna_coverages = _exact_coverage_curve_genome(genome, log_ptr, db=db)

    # Pre-normalize WGS coverage
    wgs_locations = np.arange(0, 1, 1 / size)
    wgs_coverages = _exact_coverage_curve(log_ptr=log_ptr, size=1, oor=oor, locations=wgs_locations)

    return (rna_locations, rna_coverages), (wgs_locations, wgs_coverages)


def _sample_from_system(
    genome=None, rna_positions=None, wgs_probs=None, log_ptr=None, multiplier=1, read_size=300, db=None, perfect=False
):
    """Given predicted 16S positions and PTRs, sample from that system"""

    db = RnaDB() if db is None else db
    log_ptr = np.array(log_ptr).flatten()

    # Input validation
    if (genome is None or log_ptr is None) and (rna_positions is None or wgs_probs is None):
        raise ValueError("Must provide either (genome and log_ptr) or (rna_positions and wgs_probs)")

    # We can get positions/probabilities ourselves:
    elif wgs_probs is None or rna_positions is None:
        (rna_positions, _), (_, wgs_probs) = _coverage_16s_and_wgs(genome=genome, log_ptr=log_ptr, db=db)

    # We know what happens if coverage is 0: no hits
    if multiplier == 0:
        return np.zeros_like(wgs_probs), np.zeros_like(rna_positions)

    # Sampled/expected WGS reads using Poisson distribution:
    genome_size = len(wgs_probs)
    rna_indices = (rna_positions * genome_size).astype(np.int32)
    lam = wgs_probs * multiplier
    rna_hits = np.zeros(shape=(len(rna_indices), len(log_ptr)))
    read_starts = lam if perfect else np.random.poisson(lam=lam)

    # Figure out which hits overlap 16S RNAs:
    for i, rna_index in enumerate(rna_indices):
        rna_hits[i, :] = read_starts[rna_index : rna_index + read_size].sum(axis=0)

    return read_starts, rna_hits


def _read(seq, start, length, qual="I", input_path="input"):
    """Given a sequence, return a read"""
    read = seq[start : start + length]
    return f"@{input_path}:{start}:{start+length}\n{read}\n+\n{qual*length}"


def _generate_fastq_reads(starts, input_path, qual="I", length=300, downsample=1):
    # Read sequence
    if input_path.endswith("gz"):
        with gzip.open(input_path, "rt") as handle:
            sequence = SeqIO.parse(handle, "fasta").__next__().seq
    else:
        sequence = SeqIO.parse(input_path, "fasta").__next__().seq

    # Coerce starts to integer array
    starts = np.array(starts, dtype=np.int32)

    # Add circularity and make lowercase
    sequence = f"{sequence}{sequence[:length]}".lower()

    # Transform starts variable from a count vector to a list of starts
    seqlen = len(starts)
    indices = np.arange(seqlen, dtype=np.int32)
    starts = np.repeat(indices, starts[:, 0])

    # Downsample without looping
    starts = starts[np.random.rand(len(starts)) < downsample]
    rc_random = np.random.rand(len(starts))
    starts_fwd = starts[rc_random < 0.5]
    starts_rev = starts[rc_random >= 0.5]

    reads = []
    for starts_set, seq in zip((starts_fwd, starts_rev), (sequence, rc(sequence.upper()))):
        reads += [_read(seq, start=start, length=length, qual=qual, input_path=input_path) for start in starts_set]

    return reads


def _generate_otu_table(rna_hits, genome, db=None):
    """Generate OTU table form an array of 16S hits and a genome ID"""
    db = RnaDB() if db is None else db
    rna_hits = np.array(rna_hits).flatten()

    _, md5s, gene_to_seq = db.generate_genome_objects(genome)
    return pd.Series(data=rna_hits @ gene_to_seq, index=md5s)


def simulate_samples(
    abundances: pd.DataFrame,
    log_ptrs: pd.DataFrame,
    fasta_dir: str = None,
    fasta_ext: str = ".fna.gz",
    fastq_out_path: str = None,
    db: RnaDB = None,
    multiplier: float = 1,
    perfect: bool = False,
    read_size: int = 300,
    downsample: float = 1,
    shuffle: bool = True,
) -> pd.DataFrame:
    """Fully simulate n samples for a genome. Skips WGS if paths are not given."""
    db = RnaDB() if db is None else db
    abundances = abundances.reindex(index=log_ptrs.index, columns=log_ptrs.columns)

    out = []
    for sample in log_ptrs.columns:
        sample_out = []
        sample_fastq = []
        for genome in log_ptrs.index:
            log_ptr = float(log_ptrs.loc[genome, sample])

            starts, rna_hits = _sample_from_system(
                genome=genome,
                log_ptr=log_ptr,
                db=db,
                read_size=read_size,
                multiplier=multiplier * abundances.loc[genome, sample],
                perfect=perfect,
            )
            if fasta_dir is not None and fastq_out_path is not None:
                genome_reads = _generate_fastq_reads(
                    starts=starts,
                    input_path=f"{fasta_dir}/{genome}{fasta_ext}",
                    length=read_size,
                    downsample=downsample,
                )
                sample_fastq.extend(genome_reads)
            sample_out.append(_generate_otu_table(rna_hits=rna_hits, genome=genome, db=db))
        if fasta_dir is not None and fastq_out_path is not None:
            np.random.shuffle(sample_fastq) if shuffle else None
            with open(f"{fastq_out_path}/Sample_{sample}.fastq", "w") as handle:
                print(f"Writing Sample_{sample}.fastq")
                handle.write("\n".join(sample_fastq))

        out.append(pd.DataFrame(sample_out).sum(axis=0))
    return pd.DataFrame(out).T


def make_tables(
    n_genomes=10, n_samples=20, db=None, sparsity=0.5, **kwargs
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Convenience function to quickly generate OTU tables with ground truth."""
    db = RnaDB() if db is None else db
    genomes = np.random.choice(db.complete_genomes, n_genomes, replace=False)
    samples = list(range(n_samples))

    # Vectorized generator
    T = len(genomes)
    S = len(samples)
    mask = np.random.rand(T, S) > sparsity
    log_ptrs = pd.DataFrame(index=genomes, columns=samples, data=np.log(2) * mask * np.random.rand(T, S), dtype=float)
    # log2 ensures exp(log_ptrs) stays in [0, 2]

    abundances = pd.DataFrame(
        index=genomes, columns=samples, data=mask * np.random.exponential(size=(T, S)), dtype=float
    )

    otus = simulate_samples(abundances=abundances, log_ptrs=log_ptrs, db=db, **kwargs)
    return abundances, log_ptrs, otus
