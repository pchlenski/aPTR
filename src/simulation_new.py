"""There were problems with the old simulation code, so I wrote some more concise stuff instead."""

import numpy as np
import pandas as pd
from Bio import SeqIO
from typing import Iterable, List, Tuple
import gzip
from src.database import RnaDB
from src.oor_distance import oor_distance


def _exact_coverage_curve(
    log_ptr, distances=None, oor=None, locations=None, size=1
):
    # Some Respect constraints on distances:
    if distances is None:
        # Can infer distances from OOR and locations
        if locations is not None and oor is not None:
            if np.max(locations) > size or oor > size:
                raise ValueError("Size must be larger than locations and oor")
            if np.min(locations) < 0 or oor < 0:
                raise ValueError("Location and oor can not be < 0")
            oor_distances = oor_distance(locations, oor, size)
        else:
            raise ValueError("Must provide locations, size, and OOR")

    # Enforce normalization
    elif np.max(distances) > 1 or np.min(distances) < 0:
        raise ValueError("OOR distances must be normalized.")
    else:
        oor_distances = distances

    return np.exp(1 - log_ptr * oor_distances)


def _exact_coverage_curve_genome(genome, log_ptr, db=None, wgs=False):
    """Given a genome ID and a log-PTR, generate coverages at 16S positions"""
    if db is None:
        db = RnaDB()

    # Input validation
    genome = str(genome)
    log_ptr = float(log_ptr)
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
    log_ptr = float(log_ptr)
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
):
    """Given predicted 16S use RNRPM and WGS use RP, sample from that system"""

    if db is None:
        db = RnaDB()

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

    # Sample WGS reads using Poisson distribution
    lam = wgs_probs * multiplier
    read_starts = np.random.poisson(lam=lam)

    # Figure out which hits overlap 16S RNAs:
    genome_size = len(wgs_probs)
    rna_indices = (rna_positions * genome_size).astype(int)
    rna_hits = np.zeros_like(rna_indices)
    for i, rna_index in enumerate(rna_indices):
        rna_hits[i] = read_starts[rna_index : rna_index + read_size].sum()

    return read_starts, rna_hits


def _rc(seq):
    """Reverse complement of a DNA sequence. Assumes lowercase"""
    seq = seq.lower()
    return seq.translate(str.maketrans("acgt", "tgca"))[::-1]


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
            read = _rc(read.upper())
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

    _, md5s, gene_to_seq = db.generate_genome_objects(genome)
    return pd.Series(data=rna_hits @ gene_to_seq, index=md5s)


def simulate_samples(
    abundances: pd.DataFrame,
    log_ptrs: pd.DataFrame,
    fasta_dir: str = None,
    fasta_ext: str = ".fna.gz",
    fastq_out_path: str = None,
    db: RnaDB = None,
    read_size: int = 300,
) -> pd.DataFrame:
    """Fully simulate n samples for a genome. Skips WGS if paths are not given."""
    if db is None:
        db = RnaDB()

    # Ensure index and columns match for abundances and log_ptrs
    abundances = abundances.reindex(
        index=log_ptrs.index, columns=log_ptrs.columns
    )
    abundances = abundances.fillna(0)

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
                multiplier=abundances.loc[genome, sample],
            )
            if fasta_dir is not None and fastq_out_path is not None:
                _generate_fastq_reads(
                    start=starts,
                    genome=genome,
                    input_path=f"{fasta_path}/{genome}{fasta_ext}",
                    output_path=fastq_out_path,
                )
            sample_out.append(
                _generate_otu_table(rna_hits=rna_hits, genome=genome, db=db)
            )
        out.append(pd.DataFrame(sample_out).sum(axis=0))
    return pd.DataFrame(out).T


def make_tables(
    n_genomes=10, n_samples=20, db=None, sparsity=0.5
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Convenience function to quickly generate OTU tables with ground truth."""

    if db is None:
        db = RnaDB()

    genomes = np.random.choice(db.complete_genomes, n_genomes, replace=False)
    samples = list(range(n_samples))

    log_ptrs = pd.DataFrame(index=genomes, columns=samples)
    abundances = pd.DataFrame(index=genomes, columns=samples)

    for sample in samples:
        for genome in genomes:
            if np.random.rand() < sparsity:
                log_ptrs.loc[genome, sample] = np.random.rand()
                abundances.loc[genome, sample] = np.random.lognormal(0)
            else:
                log_ptrs.loc[genome, sample] = np.nan
                abundances.loc[genome, sample] = 0

    otus = simulate_samples(abundances=abundances, log_ptrs=log_ptrs, db=db)
    return abundances, log_ptrs, otus
