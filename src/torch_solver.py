""" Class for solving OTU matrix --- Pytorch version"""

# import numpy as np
import torch
import numpy as np
from typing import List, Dict, Tuple, Callable


class TorchSolver(torch.nn.Module):
    def __init__(self, genomes, abundances, ptrs, coverages):
        super(TorchSolver, self).__init__()

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
        if abundances is not None and ptrs is not None:
            self.set_coverages(
                abundances=torch.tensor(abundances), ptrs=torch.tensor(ptrs)
            )
        else:
            self.ptrs = None
            self.abundances = None
            if coverages is not None:
                self.set_coverages(
                    coverages=torch.tensor(coverages, dtype=torch.float32)
                )
            else:
                self.coverages = None

        # Other attributes used during prediction
        self.a_hat = torch.rand(size=(self.n,))
        self.b_hat = torch.log(1 + torch.rand(size=(self.n,)))

    def forward(
        self, abundances: torch.tensor, ptrs: torch.tensor
    ) -> torch.tensor:
        """Compute convolved coverage vector (= observed coverages)"""
        a = abundances
        b = ptrs
        C = self.members
        D = self.dists
        g = a @ C + 1 - b @ D
        E = self.gene_to_seq
        return torch.exp(g) @ E


class TorchSolver_deprecated:
    """Uses gradient descent and known 16S positions to estimate abundance and PTR from a set of coverages."""

    def __init__(
        self,
        genomes: List[Dict[str, list]],
        abundances: torch.tensor = None,
        ptrs: torch.tensor = None,
        coverages: torch.tensor = None,
    ) -> None:
        """Initialize 16S system. Assumes 'genomes' is a list of dicts keyed by 'pos' and 'seqs'"""

        # Save genome info, just in case, + get matrix sizes
        self.genomes = genomes
        self.seqs = set().union(*[set(genome["seqs"]) for genome in genomes])
        self.n = len(genomes)
        self.m = torch.sum(
            torch.tensor([len(genome["pos"]) for genome in genomes])
        )
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
        if abundances is not None and ptrs is not None:
            self.set_coverages(
                abundances=torch.tensor(abundances), ptrs=torch.tensor(ptrs)
            )
        else:
            self.ptrs = None
            self.abundances = None
            if coverages is not None:
                self.set_coverages(
                    coverages=torch.tensor(coverages, dtype=torch.float32)
                )
            else:
                self.coverages = None

        # Other attributes used during prediction
        self.a_hat = None
        self.b_hat = None
        self.best_loss = None

    def _g(self, abundances: torch.tensor, ptrs: torch.tensor) -> torch.tensor:
        """
        Compute the unconvolved log-coverage vector

        g = aC + 1 - bD

        a: log-abundances
        C: membership matrix
        b: log-PTRs
        D: distance matrix
        """

        a = abundances
        b = ptrs
        C = self.members
        D = self.dists
        return a @ C + 1 - b @ D

    def forward(
        self, abundances: torch.tensor, ptrs: torch.tensor
    ) -> torch.tensor:
        """
        Compute convolved coverage vector (= observed coverages)

        f = exp(g)E = exp(aC + 1 - bD)E

        a: log-abundances
        C: membership matrix
        b: log-PTRs
        D: distance matrix
        E: sequence-sharing matrix
        """

        g = self._g(abundances, ptrs)
        E = self.gene_to_seq
        return torch.exp(g) @ E

    def set_coverages(
        self,
        abundances: torch.tensor = None,
        ptrs: torch.tensor = None,
        coverages: torch.tensor = None,
    ) -> None:
        """Set log-abundances and log-PTRs and/or true coverages for a system"""

        if abundances is not None and ptrs is not None:
            self.abundances = torch.tensor(abundances)
            self.ptrs = torch.tensor(ptrs)
            self.coverages = self.compute_coverages(abundances, ptrs)
        else:
            self.coverages = coverages

    # def loss(
    #     self, abundances: torch.tensor = None, ptrs: torch.tensor = None
    # ) -> float:
    #     """Compute the MSE between empirical and predicted coverages"""

    #     # Use a_hat, b_hat
    #     if abundances is None:
    #         abundances = self.a_hat
    #     if ptrs is None:
    #         ptrs = self.b_hat

    #     # Compute loss if coverages are known
    #     if self.coverages is not None:
    #         f_hat = self.compute_coverages(abundances, ptrs)
    #         # return torch.mean((f_hat - self.coverages) ** 2)
    #         return torch.nn.MSELoss()(f_hat, self.coverages)
    #     else:
    #         raise Exception("No known coverages computed!")

    # def gradients(
    #     self, abundances: torch.tensor = None, ptrs: torch.tensor = None
    # ) -> Tuple[torch.tensor, torch.tensor]:
    #     """
    #     Compute gradients of the loss function w/r/t log-abundances, log-PTRs.

    #     L = MSE(f_predicted, f_observed)
    #     dL/dg = 2/k exp(g) E(f_predicted - f_observed)
    #     dL/da = CdL/dg
    #     dL/db = -Ddl/dg

    #     C: membership matrix
    #     D: distance matrix
    #     E: sequence-sharing matrix
    #     f: convolved coverage vector
    #     g: unconvolved log-coverage vector
    #     """

    #     # Use a_hat and b_hat as needed
    #     if abundances is None:
    #         abundances = self.a_hat
    #     if ptrs is None:
    #         ptrs = self.b_hat

    #     # Get relevant matrices
    #     C = self.members
    #     D = self.dists
    #     E = self.gene_to_seq
    #     f_hat = self.compute_coverages(abundances, ptrs)
    #     g = self._g(abundances, ptrs)

    #     # Backprop computations
    #     dL_df = f_hat - self.coverages
    #     dL_dg = (2 / self.k) * torch.exp(g) * (E @ dL_df)
    #     dL_da = C @ dL_dg
    #     dL_db = -D @ dL_dg

    #     return dL_da, dL_db

    # def guess(
    #     self, abundances: torch.tensor = None, ptrs: torch.tensor = None
    # ) -> None:
    #     """Set an initial set of params"""

    #     # A reasonable automatic guess
    #     if abundances is None:
    #         abundances = torch.rand(self.n)
    #     if ptrs is None:
    #         ptrs = torch.log(1 + torch.rand(self.n))

    #     # Save changes as system attributes
    #     self.a_hat = abundances
    #     self.b_hat = ptrs

    # def training_step(self, lr: float = 0.0001) -> float:
    #     """One training step"""

    #     loss = self.loss(self.a_hat, self.b_hat)
    #     da, db = self.gradients(self.a_hat, self.b_hat)
    #     self.a_hat -= lr * da
    #     self.b_hat -= lr * db

    #     return loss

    # def train(
    #     self,
    #     # lr: float = 0.0001,
    #     lr_scheduler: Callable[[int], float] = lambda x: 0.0001,
    #     tolerance: float = 0.001,
    #     frequency: int = 1000,
    #     max_steps: int = torch.inf,
    #     verbose: bool = False,
    # ) -> int:
    #     """Learn log-abundances, log-PTRs until convergence in loss"""

    #     self.guess()
    #     i = 0
    #     loss_diff = torch.inf
    #     last_loss = torch.inf
    #     while loss_diff > tolerance and i < max_steps:
    #         i += 1
    #         # lr = lr_scheduler(i)
    #         # loss = self.training_step(lr=lr)
    #         loss = self.training_step(lr=)
    #         if i % frequency == 0:
    #             loss_diff = last_loss - loss
    #             last_loss = loss
    #             if verbose:
    #                 print(f"{i}\t{last_loss}\t{loss_diff}")

    #     # Save some information about this round of training
    #     self.best_loss = last_loss
    #     return i

    def train(self, n_epochs=1000, steps_per_print=100, **kwargs):
        """Use Adam to apply gradients to model parameters"""

        # Guess initial abundance and PTR
        self.a_hat = torch.rand(size=(self.n,))
        self.b_hat = torch.log(1 + torch.rand(size=(self.n,)))

        # Use Adam to optimize
        optimizer = torch.optim.Adam([self.a_hat, self.b_hat], lr=0.1)
        for epoch in range(n_epochs):
            optimizer.zero_grad()
            output = self.forward(self.a_hat, self.b_hat)
            loss = torch.nn.MSELoss()(output, self.coverages)
            loss.requires_grad = True
            loss.backward()
            optimizer.step()
            if epoch % steps_per_print == 0:
                print(f"{epoch}\t{loss}")
