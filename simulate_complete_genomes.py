"""
Use this script to simulate reads from a bunch of samples.
Only need to edit global variables at the top of the script.
"""

from uuid import uuid4
import os

import pandas as pd
import numpy as np

from src.database import RnaDB
from src.simulation import simulate_from_ids, write_output

# Edit these variables only
N_SAMPLES = 10
N_GENOMES_PER_SAMPLE = 10
SCALE = 1e6
DD = "./data/"
OUTDIR = "./experiments/simulated_complete/"
COMPLETE_ONLY = True  # Must be true for now

# Simulation code
full_db = RnaDB(
    path_to_dnaA=f"{DD}allDnaA.tsv",
    path_to_16s=f"{DD}allSSU.tsv",
)
db = full_db.db

if COMPLETE_ONLY:
    genomes = db[db["n_contigs"] == 1]["genome"].unique()
else:
    genomes = db["genome"].unique()
    raise NotImplementedError("Simulation from draft genomes not implemented")

# Unique output directory
path = f"{OUTDIR}/{uuid4()}"
os.mkdir(path)
os.mkdir(f"{path}/reads")

# Initialize PTR and coverage df
ptr_df = pd.DataFrame(columns=["genome", "sample", "ptr"])
cov_df = pd.DataFrame(columns=["genome", "sample", "reads"])
otu_df = pd.DataFrame(columns=["otu", "sample", "reads"])

for sample in range(N_SAMPLES):
    sample_genomes = np.random.choice(
        genomes, N_GENOMES_PER_SAMPLE, replace=False
    )

    samples, ptrs, coverages, otus = simulate_from_ids(
        db=db,
        ids=sample_genomes,
        fasta_path=f"{DD}seqs",
        n_samples=1,
        scale=SCALE,
        shuffle=False,  # Suppress shuffling to conserve memory
    )

    write_output(samples=samples, path=f"{path}/reads/S_{sample}", prefix="")

    # Build up larger PTR/coverage dataframe
    for genome, ptr, coverage in zip(sample_genomes, ptrs, coverages):
        ptr_df = ptr_df.append(
            {"genome": genome, "sample": sample, "ptr": ptr[0]},
            ignore_index=True,
        )
        cov_df = cov_df.append(
            {"genome": genome, "sample": sample, "reads": coverage[0]},
            ignore_index=True,
        )

    # Build up OTU dataframe
    for otu_sample in otus:
        for otu in otu_sample:
            otu_df = otu_df.append(
                {"otu": otu, "sample": sample, "reads": otu_sample[otu]},
                ignore_index=True,
            )

# Save PTR/coverage dataframes once
ptr_df.to_pickle(f"{path}/ptr.pkl")
cov_df.to_pickle(f"{path}/cov.pkl")
otu_df.to_pickle(f"{path}/otu.pkl")

# Generate tables
ptr_df.pivot("genome", "sample", "ptr").to_csv(f"{path}/ptrs.tsv", sep="\t")
cov_df.pivot("genome", "sample", "reads").to_csv(
    f"{path}/coverages.tsv", sep="\t"
)
otu_df.pivot("otu", "sample", "reads").to_csv(f"{path}/otu_table.tsv", sep="\t")
