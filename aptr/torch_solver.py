""" Class for solving OTU matrix --- Pytorch version"""

# import numpy as np
import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Callable
from aptr.database import RnaDB


class TorchSolver(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.set_vals(**kwargs)

    def set_vals(
        self,
        otus: pd.DataFrame,
        genome_ids: List[str] = None,
        md5s: List[str] = None,
        abundances: pd.DataFrame = None,
        ptrs: pd.DataFrame = None,
        normalize: bool = False,
        db: RnaDB = None,
    ):
        if db is None:
            db = RnaDB()

        # Get genomes by md5s if not provided
        if genome_ids is None:
            if md5s is None:
                print("Using OTU index for md5s")
                md5s = list(otus.index)

            genome_ids = db.find_genomes_by_md5(md5s, strict=True)
            md5s = db[genome_ids]["md5"].unique()
            # This ensures we throw out any spurious md5 sequences we may have
            # because of the reindexing by md5s later on

            # genomes = []
            # # Verify genome md5s are a subset of provided md5s
            # for genome in genome_candidates:
            #     genome_md5s = db[genome]["md5"].unique()
            #     if all([md5 in md5s for md5 in genome_md5s]):
            #         genomes.append(genome)

        if len(genome_ids) == 0:
            raise ValueError("No genomes found")

        if isinstance(otus, pd.Series):
            otus = otus.to_frame()

        # In case genomes is a list of IDs; overwrites MD5s
        genome_objs, md5s, gene_to_seq = db.generate_genome_objects(genome_ids)
        otus = otus.reindex(md5s)
        otus = otus[otus.columns[otus.sum(axis=0) > 0]]  # Remove empty samples

        # Set a bunch of attribute values for use later
        self.db = db
        self.genome_ids = genome_ids
        self.sample_ids = list(otus.columns)
        self.genome_objects = genome_objs
        self.md5s = md5s
        self.otu_table = otus
        self.normalize = normalize

        # All sizes
        self.S = otus.shape[1]
        self.T = len(genome_ids)
        self.G = np.sum([len(g["pos"]) for g in genome_objs])
        self.N = len(md5s)

        # Compute membership(M), distance (D) and sequence-sharng (S) matrices
        self.members = torch.zeros(size=(self.G, self.T))
        self.dists = torch.zeros(size=(self.G, self.T))
        self.gene_to_seq = torch.tensor(gene_to_seq.T, dtype=torch.float32)
        i = 0

        for g, genome in enumerate(genome_objs):
            pos = genome["pos"].flatten()
            j = i + len(pos)

            # Put indicator, position in correct row (g)
            self.members[i:j, g] = 1
            self.dists[i:j, g] = torch.tensor(pos)

            i = j

        # Compute coverages, etc
        self.coverages = torch.tensor(otus.values, dtype=torch.float32)
        self.coverages = torch.nan_to_num(self.coverages, nan=0)
        if self.normalize:
            self.coverages /= torch.sum(self.coverages, axis=0, keepdim=True)

        self.abundances = abundances
        self.ptrs = ptrs

    def forward(self, A, R, bias=None) -> torch.Tensor:
        """
        Compute convolved coverage vector (= observed coverages)

        Assumes the following:
        A = abundance (T x S)
        R = log-ptr (T x S)
        """
        if bias is None:
            bias = self.bias

        M = self.members
        D = self.dists
        S = self.gene_to_seq
        f = torch.diag(bias)

        Y_unconvolved = M @ A * torch.exp(-D @ R)

        return f @ (S @ Y_unconvolved)

    def train(
        self,
        lr=1e-3,
        epochs: int = 500,
        iterations: int = 1000,
        tolerance: int = 5,
        loss_fn: Callable = torch.nn.PoissonNLLLoss(log_input=False),
        A_hat: torch.Tensor = None,
        R_hat: torch.Tensor = None,
        model_bias: bool = True,
        alpha1: float = 0,
        alpha2: float = 0,
        verbose: bool = True,
        clip: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, List[float]]:
        """Initialize and train with SGD + Adam"""

        # Initialize A_hat and R_hat
        if A_hat is None:
            A_hat = torch.ones(size=(self.T, self.S), requires_grad=True)
        elif type(A_hat) is np.ndarray:
            A_hat = A_hat.reshape(self.T, self.S)
            A_hat = torch.from_numpy(A_hat).float().requires_grad_(True)
        if R_hat is None:
            R_hat = torch.ones(size=(self.T, self.S), requires_grad=True)
        elif type(R_hat) is np.ndarray:
            R_hat = R_hat.reshape(self.T, self.S)
            R_hat = torch.from_numpy(R_hat).float().requires_grad_(True)
        self.A_hat = A_hat
        self.R_hat = R_hat
        self.bias = torch.ones(self.N, requires_grad=model_bias)

        # Initialize optimizer and counters
        optimizer = torch.optim.Adam([self.A_hat, self.R_hat, self.bias], lr=lr)
        best_loss, best_A_hat, best_R_hat = torch.inf, None, None
        early_stop_counter = 0
        losses = []

        # Add as attributes
        self.optimizer = optimizer
        self.loss_fn = loss_fn

        if verbose:
            print(f"Initial:\t {loss_fn(self(self.A_hat, self.R_hat, self.bias), self.coverages)}")

        Y = self.coverages
        for epoch in range(epochs):
            for _ in range(iterations):
                optimizer.zero_grad()

                # Forward pass
                Y_hat = self(self.A_hat, self.R_hat, self.bias)
                if self.normalize:
                    Y_hat = Y_hat / Y_hat.sum(axis=0, keepdims=True)  # normalize
                loss = loss_fn(Y_hat, Y)

                # Regularize: L1 for A, L2 for R
                loss += alpha1 * torch.norm(self.A_hat, p=1)
                loss += alpha2 * torch.norm(self.R_hat, p=2)
                losses.append(loss.item())
                loss.backward()
                optimizer.step()

                # Ensure reasonable PTR - e is generally enough
                if clip:
                    self.R_hat.data = self.R_hat.clamp(0, 1)

                self.A_hat.data = self.A_hat.clamp(0, None)  # Nonneg.
                self.bias.data = self.bias.clamp(0, None)  # Nonneg.

            if verbose:
                print(f"Epoch {epoch}:\t {loss}")

            # Early stopping, per-epoch
            if best_loss > loss:
                best_loss, best_A_hat, best_R_hat = (
                    loss,
                    self.A_hat.clone(),
                    self.R_hat.clone(),
                )
                early_stop_counter = 0
            else:
                early_stop_counter += 1

            if early_stop_counter > tolerance:
                break

        return (
            best_A_hat.detach().numpy(),
            best_R_hat.detach().numpy(),
            losses,
        )
