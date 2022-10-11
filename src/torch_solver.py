""" Class for solving OTU matrix --- Pytorch version"""

# import numpy as np
import torch
import numpy as np
from typing import List, Dict, Tuple, Callable
from .database import RnaDB


class TorchSolver(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.set_vals(**kwargs)

    def set_vals(
        self, genomes, coverages, abundances=None, ptrs=None, normalize=True
    ):
        self.genomes = genomes
        self.seqs = set().union(*[set(genome["seqs"]) for genome in genomes])
        self.n = len(genomes)
        self.m = np.sum([len(genome["pos"]) for genome in genomes])
        self.k = len(self.seqs)

        # Compute membership(C), distance (D) and gene_to_seq (E) matrices
        self.members = torch.zeros(size=(self.n, self.m))
        self.dists = torch.zeros(size=(self.n, self.m))
        self.gene_to_seq = torch.zeros(size=(self.m, self.k))
        i = 0

        for g, genome in enumerate(genomes):
            j = i + len(genome["pos"])

            # Put indicator, position in correct row (g)
            self.members[g, i:j] = 1
            self.dists[g, i:j] = torch.tensor(genome["pos"])

            # Keep track of sequences
            for s, seq in enumerate(genome["seqs"]):
                self.gene_to_seq[i + s, seq] = 1
            i = j

        # Compute coverages, etc
        self.coverages = torch.tensor(coverages, dtype=torch.float32)
        if normalize:
            self.coverages /= torch.sum(self.coverages)

        # # Other attributes used during prediction
        # self.a_hat = torch.rand(size=(self.n,), requires_grad=True)
        # self.b_hat = torch.rand(size=(self.n,), requires_grad=True)

    def forward(self) -> torch.Tensor:
        """
        Compute convolved coverage vector (= observed coverages)

        Assumes the following:
        a = log-abundance
        b = log-ptr
        """
        a = self.a_hat
        b = self.b_hat
        C = self.members
        D = self.dists
        g = a @ C + 1 - b @ D
        E = self.gene_to_seq
        return torch.exp(g) @ E

    def train(
        self,
        lr=1e-4,
        epochs: int = 500,
        iterations: int = 1000,
        epsilon: float = 1e-1,
        tolerance: int = 5,
        a_hat: torch.Tensor = None,
        b_hat: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[float]]:
        """Initialize and train with SGD + Adam"""

        # Initialize a_hat and b_hat
        if a_hat is None:
            a_hat = torch.rand(size=(self.n,), requires_grad=True)
        elif type(a_hat) is np.ndarray:
            a_hat = torch.from_numpy(a_hat).float().requires_grad_(True)

        if b_hat is None:
            b_hat = torch.rand(size=(self.n,), requires_grad=True)
        elif type(b_hat) is np.ndarray:
            b_hat = torch.from_numpy(b_hat).float().requires_grad_(True)

        self.a_hat = a_hat
        self.b_hat = b_hat

        optimizer = torch.optim.Adam([self.a_hat, self.b_hat], lr=lr)

        best_loss, best_a_hat, best_b_hat = torch.inf, None, None
        early_stop_counter = 0
        losses = []

        for epoch in range(epochs):
            for _ in range(iterations):
                # Updates
                optimizer.zero_grad()
                f = self()
                loss = torch.nn.functional.mse_loss(f, self.coverages)
                losses.append(loss.item())
                loss.backward(retain_graph=True)
                optimizer.step()

                # Ensure reasonable PTR
                with torch.no_grad():
                    self.b_hat = torch.clip(b_hat, 0, 2)  # ~7.8 is plenty

            print(f"Epoch {epoch}:\t {loss}")
            # Early stopping, per-epoch
            if best_loss - loss > epsilon:
                best_loss, best_a_hat, best_b_hat = (
                    loss,
                    self.a_hat.clone(),
                    self.b_hat.clone(),
                )
                early_stop_counter = 0
            else:
                early_stop_counter += 1

            if early_stop_counter > tolerance:
                break

        return (
            best_a_hat.detach(),
            best_b_hat.detach(),
            losses,
        )


def solve_table(otus, genome_ids, db=RnaDB(), ptrs=None, abundances=None):
    """Solve an entire OTUtable"""

    # Need extra sanity checks if PTRs and abundances are given
    if abundances is not None and ptrs is not None:
        if abundances.shape != ptrs.shape:
            raise Exception("abundances should have same shape as ptrs")
        if abundances.shape[1] != otus.shape[1]:
            raise Exception("number of samples is mismatched")

    # Initialize solver system
    n_samples = otus.shape[1]
    genomes, _ = db.generate_genome_objects(genome_ids)
    solutions = []
    for i in range(n_samples):
        solver = TorchSolver(genomes=genomes, coverages=otus.iloc[:, i])
        a, b, losses = solver.train()
        solutions.append((a, b, losses))
    return solutions
