from src.tests.simulation_new_tests import *

if __name__ == "__main__":
    assert test_oor_dists()
    assert test_exact_coverage()
    assert test_exact_coverage_genome()
    assert test_coverage_16s_and_wgs()
    assert test_sample_from_system()
    assert test_rc()
    assert test_generate_fastq_reads()
    assert test_generate_otu_table()
    assert test_simulate_samples()
    assert test_make_tables()
