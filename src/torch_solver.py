""" Class for solving OTU matrix --- Pytorch version"""

# import numpy as np
import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Callable
from .database import RnaDB


class TorchSolver(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.set_vals(**kwargs)

    def set_vals(
        self, genomes, coverages, abundances=None, ptrs=None, normalize=True
    ):
        if coverages.ndim == 1:
            coverages = coverages.reshape(1, -1)
        if abundances is not None and abundances.ndim == 1:
            abundances = abundances.reshape(1, -1)
        if ptrs is not None and ptrs.ndim == 1:
            ptrs = ptrs.reshape(1, -1)

        # TODO: handle case where "genomes" is a list of IDs
        self.genomes = genomes
        self.seqs = set().union(*[set(genome["seqs"]) for genome in genomes])
        self.s = coverages.shape[0]
        self.n = len(genomes)
        self.m = np.sum([len(genome["pos"]) for genome in genomes])
        self.k = len(self.seqs)

        # Compute membership(C), distance (D) and gene_to_seq (E) matrices
        self.members = torch.zeros(size=(self.n, self.m))
        self.dists = torch.zeros(size=(self.n, self.m))
        self.gene_to_seq = torch.zeros(size=(self.m, self.k))
        i = 0

        for g, genome in enumerate(genomes):
            pos = genome["pos"].flatten()
            j = i + len(pos)

            # Put indicator, position in correct row (g)
            self.members[g, i:j] = 1
            self.dists[g, i:j] = torch.tensor(pos)

            # Keep track of sequences
            for s, seq in enumerate(genome["seqs"]):
                self.gene_to_seq[i + s, seq] = 1
            i = j

        # Compute coverages, etc
        self.coverages = torch.tensor(coverages, dtype=torch.float32)
        self.coverages = torch.nan_to_num(self.coverages, nan=0)
        if normalize:
            self.coverages /= torch.sum(self.coverages)

        # # Other attributes used during prediction
        # self.a_hat = torch.rand(size=(self.n,), requires_grad=True)
        # self.b_hat = torch.rand(size=(self.n,), requires_grad=True)

    # def forward(self, a, b) -> torch.Tensor:
    def forward(self, A, B) -> torch.Tensor:
        """
        Compute convolved coverage vector (= observed coverages)

        Assumes the following:
        a = log-abundance
        b = log-ptr
        """
        C = self.members
        D = self.dists
        # g = a @ C + 1 - b @ D
        G = A @ C + 1 - B @ D
        E = self.gene_to_seq
        # return torch.exp(g) @ E
        return torch.exp(G) @ E

    def train(
        self,
        lr=1e-3,
        epochs: int = 500,
        iterations: int = 1000,
        epsilon: float = 1e-1,
        tolerance: int = 5,
        loss_fn: Callable = torch.nn.functional.mse_loss,
        a_hat: torch.Tensor = None,
        b_hat: torch.Tensor = None,
        verbose: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, List[float]]:
        """Initialize and train with SGD + Adam"""

        # Initialize a_hat and b_hat
        if a_hat is None:
            # a_hat = torch.rand(size=(self.n,), requires_grad=True)
            a_hat = torch.zeros(size=(self.s, self.n), requires_grad=True)
        elif type(a_hat) is np.ndarray:
            a_hat = a_hat.reshape(self.s, self.n)
            a_hat = torch.from_numpy(a_hat).float().requires_grad_(True)
        if b_hat is None:
            # b_hat = torch.rand(size=(self.n,), requires_grad=True)
            b_hat = torch.zeros(size=(self.s, self.n), requires_grad=True)
        elif type(b_hat) is np.ndarray:
            b_hat = b_hat.reshape(self.s, self.n)
            b_hat = torch.from_numpy(b_hat).float().requires_grad_(True)
        self.a_hat = a_hat
        self.b_hat = b_hat

        # Initialize optimizer and counters
        optimizer = torch.optim.Adam([self.a_hat, self.b_hat], lr=lr)
        best_loss, best_a_hat, best_b_hat = torch.inf, None, None
        early_stop_counter = 0
        losses = []

        for epoch in range(epochs):
            for _ in range(iterations):
                # Updates
                f_hat = self(self.a_hat, self.b_hat)
                f_hat = f_hat / torch.sum(f_hat)  # normalize
                loss = loss_fn(f_hat, self.coverages)
                losses.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Ensure reasonable PTR - e is generally enough
                self.b_hat.data = self.b_hat.clamp(0, 1)

            if verbose:
                print(f"Epoch {epoch}:\t {loss}")

            # Early stopping, per-epoch
            if best_loss > loss:
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
            best_a_hat.detach().numpy(),
            best_b_hat.detach().numpy(),
            losses,
        )
