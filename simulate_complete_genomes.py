"""
Use this script to simulate reads from a bunch of samples.

Only need to edit three variables:
    n_samples
    n_genomes_per_sample
    scale
"""

from uuid import uuid4
import os

import util.simulation
import pandas as pd
import numpy as np

# Edit these variables only
n_samples = 100
n_genomes_per_sample = 20
scale = 1e5

# Simulation code
db = pd.read_pickle("./data/db.pkl")
complete_genomes = db[db['n_contigs'] == 1]['genome'].unique()
path = f"./out/{uuid4()}"
os.mkdir(path)

# Initialize PTR and coverage df
ptr_df = pd.DataFrame(columns=["genome", "sample", "ptr"])
cov_df = pd.DataFrame(columns=["genome", "sample", "coverage"])
otu_df = pd.DataFrame(columns=["otu", "sample", "reads"])

for sample in range(n_samples):
    genomes = np.random.choice(complete_genomes, n_genomes_per_sample)
    samples, ptrs, coverages, otus = util.simulation.simulate_from_ids(
        db=db, 
        ids=genomes, 
        fasta_path="./data/seqs", 
        n_samples=1, 
        scale=scale
    )

    util.simulation.write_output(samples=samples, path=f"{path}/S_{sample}", prefix='')

    # Build up larger PTR/coverage dataframe
    for genome, ptr, coverage in zip(genomes, ptrs, coverages):
        ptr_df = ptr_df.append(
            {"genome" : genome, "sample" : sample, "ptr" : ptr[0]},
            ignore_index=True
        )
        cov_df = cov_df.append(
            {"genome" : genome, "sample" : sample, "coverage" : coverage[0]},
            ignore_index=True
        )

    # Build up OTU dataframe
    for otu_sample in otus:
        for otu in otu_sample:
            otu_df = otu_df.append(
                {'otu' : otu, 'sample' : sample, 'reads' : otu_sample[otu]},
                ignore_index=True
            )

# Save PTR/coverage dataframes once
ptr_df.to_csv(f"{path}/ptrs.tsv", sep="\t")
cov_df.to_csv(f"{path}/coverages.tsv", sep="\t")
otu_df.to_csv(f"{path}/16s_otus.tsv", sep="\t")
