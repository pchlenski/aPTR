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
        self,
        coverages,
        genomes=None,
        md5s=None,
        abundances=None,
        ptrs=None,
        normalize=True,
        db=None,
    ):
        # Convert dataframes
        if isinstance(coverages, pd.DataFrame):
            coverages = coverages.values
        if isinstance(abundances, pd.DataFrame):
            abundances = abundances.values
        if isinstance(ptrs, pd.DataFrame):
            ptrs = ptrs.values

        # Reshape vectors to matrices
        if coverages.ndim == 1:
            coverages = coverages.reshape(1, -1)
        if abundances is not None and abundances.ndim == 1:
            abundances = abundances.reshape(1, -1)
        if ptrs is not None and ptrs.ndim == 1:
            ptrs = ptrs.reshape(1, -1)

        # Get genomes by md5s if not provided
        if genomes is None and md5s is not None:
            genomes = db.find_genomes_by_md5(md5s)
        elif genomes is None and md5s is None:
            raise ValueError("Must provide either genomes or md5s")

        # In case genomes is a list of IDs:
        if np.all([isinstance(g, str) for g in genomes]):
            if db is not None:
                genomes = [db.get_genome_objects(genomes)[0] for g in genomes]
            else:
                raise ValueError("Must provide database if passing genome IDs")

        # Set a bunch of attribute values for use later
        self.genomes = genomes
        self.seqs = set().union(*[set(genome["seqs"]) for genome in genomes])
        self.s = coverages.shape[1]
        self.n = len(genomes)
        self.m = np.sum([len(genome["pos"]) for genome in genomes])
        self.k = len(self.seqs)

        # Compute membership(C), distance (D) and gene_to_seq (E) matrices
        self.members = torch.zeros(size=(self.m, self.n))
        self.dists = torch.zeros(size=(self.m, self.n))
        self.gene_to_seq = torch.zeros(size=(self.k, self.m))
        i = 0

        for g, genome in enumerate(genomes):
            pos = genome["pos"].flatten()
            j = i + len(pos)

            # Put indicator, position in correct row (g)
            self.members[i:j, g] = 1
            self.dists[i:j, g] = torch.tensor(pos)

            # Keep track of sequences
            for s, seq in enumerate(genome["seqs"]):
                self.gene_to_seq[seq, i + s] = 1
            i = j

        # Compute coverages, etc
        self.coverages = torch.tensor(coverages, dtype=torch.float32)
        self.coverages = torch.nan_to_num(self.coverages, nan=0)
        if normalize:
            self.coverages /= torch.sum(self.coverages)

    # def forward(self, a, b) -> torch.Tensor:
    def forward(self, A, B) -> torch.Tensor:
        """
        Compute convolved coverage vector (= observed coverages)

        Assumes the following:
        A = log-abundance
        B = log-ptr
        """
        C = self.members
        D = self.dists
        G = C @ A + 1 - D @ B
        E = self.gene_to_seq
        return E @ torch.exp(G)

    def train(
        self,
        lr=1e-3,
        epochs: int = 500,
        iterations: int = 1000,
        tolerance: int = 5,
        loss_fn: Callable = torch.nn.functional.mse_loss,
        A_hat: torch.Tensor = None,
        B_hat: torch.Tensor = None,
        verbose: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, List[float]]:
        """Initialize and train with SGD + Adam"""

        # Initialize a_hat and b_hat
        if A_hat is None:
            A_hat = torch.zeros(size=(self.n, self.s), requires_grad=True)
        elif type(A_hat) is np.ndarray:
            A_hat = A_hat.reshape(self.s, self.n)
            A_hat = torch.from_numpy(A_hat).float().requires_grad_(True)
        if B_hat is None:
            B_hat = torch.zeros(size=(self.n, self.s), requires_grad=True)
        elif type(B_hat) is np.ndarray:
            B_hat = B_hat.reshape(self.n, self.s)
            B_hat = torch.from_numpy(B_hat).float().requires_grad_(True)
        self.A_hat = A_hat
        self.B_hat = B_hat

        # Initialize optimizer and counters
        optimizer = torch.optim.Adam([self.A_hat, self.B_hat], lr=lr)
        best_loss, best_A_hat, best_B_hat = torch.inf, None, None
        early_stop_counter = 0
        losses = []

        for epoch in range(epochs):
            for _ in range(iterations):
                # Updates
                F_hat = self(self.A_hat, self.B_hat)
                F_hat = F_hat / torch.sum(F_hat)  # normalize
                loss = loss_fn(F_hat, self.coverages)
                losses.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Ensure reasonable PTR - e is generally enough
                self.B_hat.data = self.B_hat.clamp(0, 1)

            if verbose:
                print(f"Epoch {epoch}:\t {loss}")

            # Early stopping, per-epoch
            if best_loss > loss:
                best_loss, best_A_hat, best_B_hat = (
                    loss,
                    self.A_hat.clone(),
                    self.B_hat.clone(),
                )
                early_stop_counter = 0
            else:
                early_stop_counter += 1

            if early_stop_counter > tolerance:
                break

        return (
            best_A_hat.detach().numpy(),
            best_B_hat.detach().numpy(),
            losses,
        )
