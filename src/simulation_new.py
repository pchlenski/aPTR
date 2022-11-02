"""There were problems with the old simulation code, so I wrote some more concise stuff instead."""

import numpy as np
import pandas as pd
from Bio import SeqIO
from typing import Iterable, List, Tuple
import gzip
from src.database import RnaDB
from src.oor_distance import oor_distance
from src.string_operations import rc


def _exact_coverage_curve(
    log_ptr, distances=None, oor=None, locations=None, size=1
):
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

    return np.exp(1 - log_ptr * oor_distances)


def _exact_coverage_curve_genome(genome, log_ptr, db=None):
    """Given a genome ID and a log-PTR, generate coverages at 16S positions"""
    if db is None:
        db = RnaDB()

    # Input validation
    genome = str(genome)
    log_ptr = np.array(log_ptr)
    if not isinstance(db, RnaDB):
        raise TypeError("db must be an RnaDB")

    # Use DB to get locations and return coverages
    size = db[genome]["size"].max()
    rna_locations = np.array(db[genome]["16s_position"] / size)
    oor = float(db[genome]["oor_position"].max() / size)

    return (
        rna_locations,
        _exact_coverage_curve(log_ptr, oor=oor, locations=rna_locations),
    )


def _coverage_16s_and_wgs(genome, log_ptr, db=None):
    """Given a genome ID and a log-PTR, simulate 16S and WGS coverage simultaneously"""
    if db is None:
        db = RnaDB()

    # Input validation
    genome = str(genome)
    log_ptr = np.array(log_ptr)
    if not isinstance(db, RnaDB):
        raise TypeError("db must be an RnaDB")

    # Find number of wgs reads from get_genome_reads:
    size = db[genome]["size"].max()
    oor = db[genome]["oor_position"].max() / size
    rna_locations, rna_coverages = _exact_coverage_curve_genome(
        genome, log_ptr, db=db
    )

    # Pre-normalize WGS coverage
    wgs_locations = np.arange(0, 1, 1 / size)
    wgs_coverages = _exact_coverage_curve(
        log_ptr=log_ptr, size=1, oor=oor, locations=wgs_locations
    )

    return (rna_locations, rna_coverages), (wgs_locations, wgs_coverages)


def _sample_from_system(
    genome=None,
    rna_positions=None,
    wgs_probs=None,
    log_ptr=None,
    multiplier=1,
    read_size=300,
    db=None,
    perfect=False,
):
    """Given predicted 16S positions and PTRs, sample from that system"""

    if db is None:
        db = RnaDB()

    log_ptr = np.array(log_ptr).flatten()

    # Input validation
    if (genome is None or log_ptr is None) and (
        rna_positions is None or wgs_probs is None
    ):
        raise ValueError(
            "Must provide either (genome and log_ptr) or (rna_positions and wgs_probs)"
        )

    # We can get positions/probabilities ourselves:
    elif wgs_probs is None or rna_positions is None:
        (rna_positions, _), (_, wgs_probs) = _coverage_16s_and_wgs(
            genome=genome, log_ptr=log_ptr, db=db
        )

    # We know what happens if coverage is 0: no hits
    if multiplier == 0:
        return np.zeros_like(wgs_probs), np.zeros_like(rna_positions)

    # Sampled/expected WGS reads using Poisson distribution:
    genome_size = len(wgs_probs)
    rna_indices = (rna_positions * genome_size).astype(int)
    lam = wgs_probs * multiplier
    rna_hits = np.zeros(shape=(len(rna_indices), len(log_ptr)))
    if perfect:
        read_starts = lam  # Use expectations
    else:
        read_starts = np.random.poisson(lam=lam)

    # Figure out which hits overlap 16S RNAs:
    for i, rna_index in enumerate(rna_indices):
        rna_hits[i, :] = read_starts[rna_index : rna_index + read_size].sum(
            axis=0
        )

    return read_starts, rna_hits


def _generate_fastq_reads(starts, input_path, output_path=None, length=300):
    # Read sequence
    if input_path.endswith("gz"):
        with gzip.open(input_path, "rt") as handle:
            sequence = SeqIO.parse(handle, "fasta").__next__().seq
    else:
        sequence = SeqIO.parse(input_path, "fasta").__next__().seq

    # Add circularity and make lowercase
    sequence = f"{sequence}{sequence[:length]}".lower()

    # Put out fastq reads
    reads = []
    for start in starts:
        read = sequence[start : start + length]
        if np.random.rand() < 0.5:
            read = rc(read.upper())
        reads.append(
            f"@{input_path}:{start}:{start+length}\n{read}\n+\n{'#'*length}"
        )

    # Write, if output path is provided:
    if output_path is not None:
        with open(output_path, "w") as handle:
            handle.write("\n".join(reads))

    return reads


def _generate_otu_table(rna_hits, genome, db=None):
    """Generate OTU table form an array of 16S hits and a genome ID"""
    if db is None:
        db = RnaDB()

    rna_hits = np.array(rna_hits).flatten()

    _, md5s, gene_to_seq = db.generate_genome_objects(genome)
    return pd.Series(data=rna_hits @ gene_to_seq, index=md5s)


def simulate_samples(
    log_abundances: pd.DataFrame,
    log_ptrs: pd.DataFrame,
    fasta_dir: str = None,
    fasta_ext: str = ".fna.gz",
    fastq_out_path: str = None,
    db: RnaDB = None,
    multiplier: float = 1,
    perfect: bool = False,
    read_size: int = 300,
) -> pd.DataFrame:
    """Fully simulate n samples for a genome. Skips WGS if paths are not given."""
    if db is None:
        db = RnaDB()

    # Ensure index and columns match for abundances and log_ptrs
    log_abundances = log_abundances.reindex(
        index=log_ptrs.index, columns=log_ptrs.columns
    )
    abundances = np.exp(log_abundances).fillna(0)

    out = []
    for sample in log_ptrs.columns:
        sample_out = []
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
                _generate_fastq_reads(
                    start=starts,
                    genome=genome,
                    input_path=f"{fasta_dir}/{genome}{fasta_ext}",
                    output_path=fastq_out_path,
                )
            sample_out.append(
                _generate_otu_table(rna_hits=rna_hits, genome=genome, db=db)
            )
        out.append(pd.DataFrame(sample_out).sum(axis=0))
    return pd.DataFrame(out).T


def make_tables(
    n_genomes=10, n_samples=20, db=None, sparsity=0.5, **kwargs
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Convenience function to quickly generate OTU tables with ground truth."""

    if db is None:
        db = RnaDB()

    genomes = np.random.choice(db.complete_genomes, n_genomes, replace=False)
    samples = list(range(n_samples))

    # Vectorized generator
    n = len(genomes)
    s = len(samples)
    mask = np.random.rand(n, s) > sparsity
    log_ptrs = pd.DataFrame(
        index=genomes,
        columns=samples,
        data=mask * np.random.rand(n, s),
        dtype=float,
    )
    log_abundances = pd.DataFrame(
        index=genomes,
        columns=samples,
        data=mask * np.random.lognormal(size=(n, s)),
        dtype=float,
    )

    otus = simulate_samples(
        log_abundances=log_abundances, log_ptrs=log_ptrs, db=db, **kwargs
    )
    return log_abundances, log_ptrs, otus
