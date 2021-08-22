import util.simulation
import pandas as pd
import numpy as np

# Edit these variables only
n_samples = 30
n_genomes_per_sample = 10

# Simulation code
db = pd.read_pickle("./data/db.pkl")
complete_genomes = db[db['n_contigs'] == 1]['genome'].unique()

for sample in range(n_samples):
    genomes = np.random.choice(complete_genomes, n_genomes_per_sample)
    reads, ptrs, coverages = util.simulation.simulate_from_ids(db, genomes, "./data/seqs", n_samples=1)

    # print(type(reads))
    # print(len(reads))
    # print(type(reads[0]))
    # print(len(reads[0]))
    # print()
    util.simulation.write_output(reads, ptrs, coverages)