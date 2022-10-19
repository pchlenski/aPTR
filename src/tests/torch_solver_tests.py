import numpy as np
import torch

from src.torch_solver import TorchSolver
from src.database import RnaDB
from src.simulation_new import make_tables


def test_init():
    return True


def test_set_vals():
    return True


def test_forward_vector():
    return True


def test_forward_tabular():
    return True


def test_train_vector():
    return True


def test_train_tabular():
    return True


def test_early_stopping():
    return True


def test_gradients():
    """
    Test that gradients agree with analytic solution:
    Letting G := exp(CA + 1 - DB), we have:
    dL/dA = 2 * C^T (G * E^T(EG - F))
    dL/dB = -2 * D^T (G * E^T(EG - F))
    Where * is element-wise multiplication and ^T is transpose.
    """
    rnadb = RnaDB(
        path_to_dnaA="./data/allDnaA.tsv",
        path_to_16s="./data/allSSU.tsv",
    )

    np.random.seed(42)
    abundances, ptrs, otus = make_tables(
        n_genomes=4, n_samples=3, sparsity=1, db=rnadb  # Dense for simplicity
    )

    # Set up solver
    solver = TorchSolver(
        genomes=rnadb.generate_genome_objects(list(abundances.index))[0],
        coverages=otus,
    )
    torch.manual_seed(42)
    solver.A_hat = torch.rand(size=abundances.shape, requires_grad=True)
    solver.B_hat = torch.rand(size=ptrs.shape, requires_grad=True)
    F_hat = solver(solver.A_hat, solver.B_hat)
    loss = torch.sum((F_hat - solver.coverages) ** 2)
    loss.backward()

    # Verify gradients match analytic form
    G = torch.exp(
        solver.members @ solver.A_hat + 1 - solver.dists @ solver.B_hat
    )
    dL_dG = 2 * G * (solver.gene_to_seq.T @ (F_hat - solver.coverages))
    assert np.allclose(
        solver.A_hat.grad.detach().numpy(),
        (solver.members.T @ dL_dG).detach().numpy(),
    )
    assert np.allclose(
        solver.B_hat.grad.detach().numpy(),
        (-solver.dists.T @ dL_dG).detach().numpy(),
    )

    # Assert shapes of gradients
    assert solver.A_hat.grad.shape == abundances.shape
    assert solver.B_hat.grad.shape == ptrs.shape

    # Assert gradients are not None
    assert solver.A_hat.grad is not None
    assert solver.B_hat.grad is not None

    return True
