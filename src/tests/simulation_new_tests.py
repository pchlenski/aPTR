import numpy as np
import pytest

from ..database import RnaDB
from ..oor_distance import oor_distance
from ..simulation_new import (
    _exact_coverage_curve,
    _exact_coverage_curve_genome,
    _coverage_16s_and_wgs,
    _sample_from_system,
    _rc,
    _generate_fastq_reads,
    _generate_otu_table,
    simulate_samples,
)


def test_oor_dists():
    # Distance should wrap around, i.e. 1==0:
    assert np.allclose(oor_distance(0, 1, 1), 0)
    assert np.allclose(oor_distance(0, 0, 1), 0)

    # Assert symmetry:
    x = np.linspace(0, 1, 100)
    assert np.allclose(
        oor_distance(x[:50], 0, 1), oor_distance(x[50:], 0, 1)[::-1]
    )

    # Symmetry around trough as well:
    assert np.allclose(
        oor_distance(x[:50], 0.5, 1), oor_distance(x[50:], 0.5, 1)[::-1]
    )

    # Test custom size and loc - random example I'm confident about:
    input_arr = [0, 2, 4, 20]
    input_oor = 3
    input_size = 100
    output_dists = oor_distance(input_arr, input_oor, input_size)
    assert np.allclose(output_dists, [0.06, 0.02, 0.02, 0.34])

    # Test with/without normalization:
    output_dists_nonnorm = oor_distance(
        input_arr, input_oor, input_size, normalized=False
    )
    # Don't forget to scale by 2
    assert np.allclose(output_dists * input_size / 2, output_dists_nonnorm)

    return True


def test_exact_coverage():
    lp = np.log(1.5)  # log-ptr

    # Verify value errors:
    # Value error 1: location greater than size:
    with pytest.raises(ValueError):
        _exact_coverage_curve(lp, locations=[0, 101], size=100)

    # Value error 2: oor greater than size:
    with pytest.raises(ValueError):
        _exact_coverage_curve(lp, oor=101, locations=[0], size=100)

    # Value error 3: locations contains a negative
    with pytest.raises(ValueError):
        _exact_coverage_curve(lp, locations=[10, -0.5, 20, 0.3], oor=6)

    # Value error 3: distance contains a value greater than 1:
    with pytest.raises(ValueError):
        _exact_coverage_curve(lp, distances=[0, 2.1], oor=1, size=100)

    # Value error 4: distance contains a value less than 0:
    with pytest.raises(ValueError):
        _exact_coverage_curve(lp, distances=[0, -0.1], oor=1, size=100)

    # Verify PTRs match empirical estimates:
    for ptr in [0.5, 1, 1.5, 2, 4, 8, 16]:
        for oor in [0, 0.2, 0.4, 0.6, 0.8, 1]:
            term = (oor + 1) % 2  # Replication terminus
            p, t = _exact_coverage_curve(
                np.log(ptr), locations=[oor, term], oor=oor, size=2
            )
            assert np.allclose(p / t, ptr)

    return True


def test_exact_coverage_genome():
    # Check error
    db = RnaDB(
        path_to_dnaA="./data/allDnaA.tsv",
        path_to_16s="./data/allSSU.tsv",
    )  # Assumes you run script from "aptr" top-level directory

    lp = np.log(1.7)
    genome = "903510.3"

    # No such genome error
    with pytest.raises(ValueError):
        _exact_coverage_curve_genome(
            "asdfdasf", np.log(1.5), db=db
        )  # incorrect genome

    # Some genome that I tried:
    rna_locations_correct = np.array(
        [
            0.1033291,
            0.05903599,
            0.21586614,
            0.01197195,
            0.07877703,
            0.16483294,
            0.66608203,
        ]
    )
    coverages_correct = np.array(
        [
            2.62183232,
            2.68886781,
            2.32668095,
            2.55786571,
            2.69104469,
            2.45616717,
            1.77197593,
        ]
    )
    rna_locations, coverages = _exact_coverage_curve_genome(genome, lp, db=db)
    assert np.allclose(rna_locations, rna_locations_correct)
    assert np.allclose(coverages, coverages_correct)

    return True


def test_coverage_16s_and_wgs():
    db = RnaDB(
        path_to_dnaA="./data/allDnaA.tsv",
        path_to_16s="./data/allSSU.tsv",
    )  # Assumes you run script from "aptr" top-level directory

    genome = "903510.3"
    (
        (rna_locations, rna_coverages),
        (wgs_locations, wgs_coverages),
    ) = _coverage_16s_and_wgs(genome, np.log(1.5), db=db)

    # Verify datatypes and shapes:
    assert isinstance(rna_locations, np.ndarray) and rna_locations.ndim == 1
    assert isinstance(rna_coverages, np.ndarray) and rna_coverages.ndim == 1
    assert isinstance(wgs_locations, np.ndarray) and wgs_locations.ndim == 1
    assert isinstance(wgs_coverages, np.ndarray) and wgs_coverages.ndim == 1

    # Verify matching
    rna_indices = (rna_locations * len(wgs_locations)).astype(int)
    assert np.allclose(rna_coverages, wgs_coverages[rna_indices])

    return True


def test_sample_from_system():
    db = RnaDB(
        path_to_dnaA="./data/allDnaA.tsv",
        path_to_16s="./data/allSSU.tsv",
    )  # Assumes you run script from "aptr" top-level directory

    gen = "903510.3"
    lp = np.log(1.6)
    # Case 1: use genome and log_ptr
    np.random.seed(42)
    read_starts1, rna_hits1 = _sample_from_system(genome=gen, log_ptr=lp, db=db)

    # Case 2: use rna_positions and wgs_probs
    (rna_positions, _), (_, wgs_probs) = _coverage_16s_and_wgs(
        genome=gen, log_ptr=lp, db=db
    )
    np.random.seed(42)
    read_starts2, rna_hits2 = _sample_from_system(
        rna_positions=rna_positions, wgs_probs=wgs_probs, db=db
    )

    # Check that samples are equivalent:
    assert np.allclose(read_starts1, read_starts2)
    assert np.allclose(rna_hits1, rna_hits2)

    return True


def test_rc():
    sequence = "ACTGACTGA"
    assert _rc(sequence) == "tcagtcagt"

    return True


def test_generate_fastq_reads():
    np.random.seed(3)  # Prevent RC
    reads = _generate_fastq_reads(
        [0, 20, 1000], "./data/seqs/903510.3.fna.gz", length=20
    )
    assert len(reads) == 3
    assert np.all([len(read.split("\n")[1]) == 20 for read in reads])
    assert reads[0].split("\n")[1] == "ataccaacctgacggcctag"
    assert reads[1].split("\n")[1] == "taggatgtgctcacgagttt"
    assert reads[2].split("\n")[1] == "accaggcgaaggctttgtct"

    return True


def test_generate_otu_table():
    db = RnaDB(
        path_to_dnaA="./data/allDnaA.tsv",
        path_to_16s="./data/allSSU.tsv",
    )  # Assumes you run script from "aptr" top-level directory

    otus = _generate_otu_table(
        [10, 100, 5, 100, 9, 1, 90], genome="903510.3", db=db
    )
    assert np.allclose(otus.values, [10, 105, 100, 9, 1, 90])
    assert list(otus.index) == [
        "f0aa7f8c0a383f20f4b3f3942871dd69",
        "77a903b4bc965f2ca9a34ccc773572e9",
        "5bc0a65936ed7d96bc02cb724e1cd5dd",
        "1b7ac45601266ed9b1f1379d931ab895",
        "8ff71171d7bc1cf309fa5360df79c3e1",
        "2b8976e8e1c5a55f9048b047d90301eb",
    ]  # Note that we go from 7 inputs to 6 outputs because of sequence sharing

    return True


def test_simulate_samples():
    db = RnaDB(
        path_to_dnaA="./data/allDnaA.tsv",
        path_to_16s="./data/allSSU.tsv",
    )  # Assumes you run script from "aptr" top-level directory

    # Case with a signle log_ptr
    np.random.seed(42)
    otus = simulate_samples(genome="903510.3", log_ptrs=np.log(1.5), db=db)
    assert otus.shape == (6, 1)
    assert np.allclose(otus[0], [790, 1521, 754, 786, 771, 567])
    assert list(otus.index) == [
        "f0aa7f8c0a383f20f4b3f3942871dd69",
        "77a903b4bc965f2ca9a34ccc773572e9",
        "5bc0a65936ed7d96bc02cb724e1cd5dd",
        "1b7ac45601266ed9b1f1379d931ab895",
        "8ff71171d7bc1cf309fa5360df79c3e1",
        "2b8976e8e1c5a55f9048b047d90301eb",
    ]

    # Case with multiple log_ptrs
    np.random.seed(42)
    otus2 = simulate_samples(
        genome="903510.3", log_ptrs=np.log([1.5, 1.6, 1]), db=db
    )
    print(otus2)
    assert otus2.shape == (6, 3)
    assert list(otus2.columns) == [0, 1, 2]

    return True
