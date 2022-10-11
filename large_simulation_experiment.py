# Factored out from what_drives_error.ipynb notebook

from src.simulation import simulate_from_ids
from src.torch_solver import TorchSolver, solve_table
from src.database import RnaDB

import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tqdm import tqdm

N_SAMPLES = 10
SCALE = 1e7

db = RnaDB(
    path_to_dnaA="./data/allDnaA.tsv",
    path_to_16s="./data/allSSU.tsv",
)

results = pd.DataFrame(
    columns=[
        "genome",
        "trial",
        "a",
        "b",
        "a_hat",
        "b_hat",
        "final_loss",
        "a_err",
        "b_err",
        "n_reads",
    ]
)

for genome in tqdm(db.complete_genomes):
    samples, ptrs, abundances, otus = simulate_from_ids(
        ids=[genome], n_samples=N_SAMPLES, scale=SCALE, db=db, verbose=False
    )
    solutions = solve_table(otus, [genome], db=db, verbose=False)
    for idx, (a_hat, b_hat, losses) in enumerate(solutions):
        a_err = np.abs(
            np.exp(a_hat[0]) - abundances[0, 0]
        )  # Not meaninful here
        b_err = np.abs(np.exp(b_hat[0]) - ptrs[0, 0])
        results = results.append(
            {
                "genome": genome,
                "trial": idx,
                "a": abundances[0, idx],
                "b": ptrs[0, idx],
                "a_hat": a_hat[0],
                "b_hat": b_hat[0],
                "final_loss": losses[-1],
                "a_err": a_err,
                "b_err": b_err,
                "n_reads": otus[idx].sum(),
            },
            ignore_index=True,
        )

results.to_csv("./data/scores/large_simulation_experiment.csv")
