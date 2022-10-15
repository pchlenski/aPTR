# # Factored out from what_drives_error.ipynb notebook

# from src.simulation import simulate_from_ids
# from src.torch_solver import TorchSolver, solve_table
# from src.database import RnaDB

# import torch
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd

# from tqdm import tqdm

# N_SAMPLES = 10
# SCALE = 1e7

# db = RnaDB(
#     path_to_dnaA="./data/allDnaA.tsv",
#     path_to_16s="./data/allSSU.tsv",
# )

# results = pd.DataFrame(
#     columns=[
#         "genome",
#         "trial",
#         "a",
#         "b",
#         "a_hat",
#         "b_hat",
#         "final_loss",
#         "a_err",
#         "b_err",
#         "n_reads",
#     ]
# )

# for genome in tqdm(db.complete_genomes):
#     samples, ptrs, abundances, otus = simulate_from_ids(
#         ids=[genome], n_samples=N_SAMPLES, scale=SCALE, db=db, verbose=False
#     )
#     solutions = solve_table(otus, [genome], db=db, verbose=False)
#     for idx, (a_hat, b_hat, losses) in enumerate(solutions):
#         a_err = np.abs(
#             np.exp(a_hat[0]) - abundances[0, 0]
#         )  # Not meaninful here
#         b_err = np.abs(np.exp(b_hat[0]) - ptrs[0, 0])
#         results = results.append(
#             {
#                 "genome": genome,
#                 "trial": idx,
#                 "a": abundances[0, idx],
#                 "b": ptrs[0, idx],
#                 "a_hat": a_hat[0],
#                 "b_hat": b_hat[0],
#                 "final_loss": losses[-1],
#                 "a_err": a_err,
#                 "b_err": b_err,
#                 "n_reads": otus[idx].sum(),
#             },
#             ignore_index=True,
#         )

# results.to_csv("./data/scores/large_simulation_experiment.csv")

import numpy as np
import pandas as pd
from tqdm import tqdm
from src.simulation_new import simulate_sample
from src.database import RnaDB
from src.torch_solver import TorchSolver

rnadb = RnaDB(
    path_to_dnaA="./data/allDnaA.tsv",
    path_to_16s="./data/allSSU.tsv",
)


def test_with_simulation(genome, log_ptr, db=rnadb):
    sample = simulate_sample(genome=genome, log_ptr=log_ptr, db=rnadb)
    solver = TorchSolver(
        genomes=db.generate_genome_objects([genome])[0], coverages=sample
    )
    a, b, l = solver.train(verbose=False, epochs=2)
    return a, b, l


results = pd.DataFrame(
    columns=["scale", "genome_id", "trial", "ptr", "est_ptr", "err", "loss"]
)

for scale in [0.001, 0.01, 0.1, 1, 10, 100]:
    print(f"Starting scale: {scale}")
    # for genome in tqdm(rnadb.complete_genomes):
    for genome in tqdm(rnadb.complete_genomes):
        for trial in range(10):
            try:
                lp = np.log(np.random.rand() + 1)
                a, b, l = test_with_simulation(genome, lp, rnadb)
                results = results.append(
                    {
                        "scale": scale,
                        "genome_id": genome,
                        "trial": trial,
                        "ptr": np.exp(lp),
                        "est_ptr": np.exp(b[0]),
                        "err": np.exp(lp) - np.exp(b[0]),
                        "loss": l[-1],
                    },
                    ignore_index=True,
                )
            except Exception as e:
                print(f"Problem with {scale} {genome} {trial}: {str(e)}")

results.to_csv(
    "./data/scores/large_simulation_experiment_newsim.csv", index=False
)
