#!/usr/bin/env python3

""" This is the command-line tool that runs the entire aPTR pipeline """

import os
import uuid
import argparse
import pandas as pd
import pickle
import numpy as np
import torch
from src.string_operations import rc
from src.preprocess_samples import preprocess_samples
from src.database import RnaDB
from src.new_filter import save_as_vsearch_db
from src.torch_solver import TorchSolver


def get_args():
    """All arguments for the aPTR pipeline"""
    parser = argparse.ArgumentParser(
        description="aPTR: a pipeline for solving metagenomic samples",
    )
    parser.add_argument(
        "path",
        type=str,
        help="Path to parent of 'reads' directory containing FASTQ files.",
    )
    parser.add_argument(
        "adapter1",
        type=str,
        help="Adapter sequence for the 3' end of the reads. Equivalent to the CUTADAPT -A/-a option.",
        default="",
    )
    parser.add_argument(
        "adapter2",
        type=str,
        help="Adapter sequence for the 5' end of the reads. Equivalent to the CUTADAPT -G/-g option.",
        default="",
    )
    parser.add_argument(
        "--db_path",
        type=str,
        help="Path to a pickled RnaDB object. Skips database generation.",
    )
    parser.add_argument(
        "--readcounts_path",
        type=str,
        default=None,
        help="Path to processed read counts. Skips preprocessing.",
    )
    parser.add_argument(
        "--otu_path",
        type=str,
        help="Path to an OTU table. Skips preprocessing.",
    )
    parser.add_argument(
        "--rc_adapter1",
        action="store_true",
        help="Reverse-complement adapter1",
    )
    parser.add_argument(
        "--rc_adapter2",
        action="store_true",
        help="Reverse-complement adapter2",
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        default=1.0,
        help="Cutoff for OTU table filtering. Default: 1.0",
    )
    parser.add_argument(
        "--min_n_reads",
        type=int,
        default=1000,
        help="Minimum number of reads per genome to return a PTR estimate.",
    )
    parser.add_argument(
        "--l1",
        type=float,
        default=0.0,
        help="L1 regularization coefficient on abundances. Default: 0.0",
    )
    parser.add_argument(
        "--l2",
        type=float,
        default=0.0,
        help="L2 regularization coefficient on PTRs. Default: 0.0",
    )
    return parser.parse_args()


def run_aptr():
    """Run the aPTR pipeline"""
    # Get arguments
    args = get_args()

    # Suppress pandas indexing warning
    pd.options.mode.chained_assignment = None

    # Reverse-complement adapters if necessary
    if args.rc_adapter1:
        args.adapter1 = rc(args.adapter1)
    if args.rc_adapter2:
        args.adapter2 = rc(args.adapter2)

    outdir = f"{args.path}/aptr_{uuid.uuid4()}"

    if args.db_path:
        db_pickle_path = args.db_path
        db = pickle.load(open(db_pickle_path, "rb"))
    else:
        db = RnaDB(
            path_to_dnaA="./data/allDnaA.tsv",
            path_to_16s="./data/allSSU.tsv",
            left_primer=args.adapter1,
            right_primer=args.adapter2,
        )
        if len(db.db) < 10:
            print("Warning: DB is very small. Perhaps try reversing a primer?")
        elif np.median([len(x) for x in db.db["16s_sequence"]]) < 100:
            print(
                "Warning: Sequences are very short. Perhaps try reversing a primer?"
            )
        db_pickle_path = f"{outdir}/db.pkl"
        db_fasta_path = f"{outdir}/db.fasta"
        try:
            os.mkdir(outdir)
        except FileExistsError:
            pass
        print(f"Output directory UUID: {outdir}")

        # Save a reduced database with adapters cut
        save_as_vsearch_db(db.db, output_file_path=db_fasta_path)
        pickle.dump(db, open(f"{outdir}/db.pkl", "wb"))
        db.db.to_csv(f"{outdir}/db.csv")

    # All the preprocessing takes place here
    if args.otu_path:
        otu_path = args.otu_path
    else:
        print("Preprocessing samples...")
        preprocess_samples(
            path=args.path,
            # adapter1=args.adapter1,
            # adapter2=args.adapter2,
            adapter1="",  # TODO: this is a temporary hack to avoid cutadapt problems
            adapter2="",  # TODO: this is a temporary hack to avoid cutadapt problems
            db_fasta_path=db_fasta_path,
            readcounts_path=args.readcounts_path,
            outdir=outdir,
            cutoff=args.cutoff,
        )
        otu_path = f"{outdir}/otu_table.tsv"

    # Infer PTRs
    otus = pd.read_table(otu_path, index_col=0)
    if len(otus) == 0:
        print("No OTUs found. Exiting. You may want to try a different cutoff.")
        return

    solver = TorchSolver(md5s=otus.index, otus=otus, db=db)
    solver.otu_table.to_csv(f"{outdir}/filtered_otu_table.tsv", sep="\t")
    pickle.dump(solver, open(f"{outdir}/solver.pkl", "wb"))
    print(f"Genomes: {solver.genome_ids}")

    solver.train(lr=0.1, tolerance=1e-6, clip=True, l1=args.l1, l2=args.l2)
    inferred_ptrs = pd.DataFrame(
        data=solver.B_hat.exp2().detach().numpy(),
        index=solver.genome_ids,
        columns=solver.sample_ids,
    )
    inferred_abundances = pd.DataFrame(
        # data=solver.A_hat.exp().detach().numpy(),
        data=solver.A_hat.detach().numpy(),
        index=solver.genome_ids,
        columns=solver.sample_ids,
    )

    # Figure out how many reads each estimate got
    unconvolved_coverages = (
        solver.dists
        @ solver.A_hat
        * torch.exp2(1 - solver.dists @ solver.B_hat)
    )
    unconvolved_coverages_normed = (
        unconvolved_coverages / unconvolved_coverages.sum(axis=0)
    )
    raw_reads = (
        unconvolved_coverages_normed.detach().numpy()
        * solver.otu_table.sum(axis=0).values
    )
    reads_per_genome = np.linalg.pinv(solver.members) @ raw_reads
    n_reads_used = pd.DataFrame(
        data=reads_per_genome,
        index=solver.genome_ids,
        columns=solver.sample_ids,
    )

    # Filter by cutoff
    mask = n_reads_used > args.min_n_reads
    # mask &= inferred_ptrs >= 1
    # mask &= inferred_ptrs <= 3
    inferred_ptrs = inferred_ptrs[mask]
    # inferred_abundances = inferred_abundances[mask]

    # Dump again after training
    pickle.dump(solver, open(f"{outdir}/solver.pkl", "wb"))

    # Save inferred quantities
    inferred_ptrs.to_csv(f"{outdir}/inferred_ptrs.tsv", sep="\t")
    inferred_abundances.to_csv(f"{outdir}/inferred_abundances.tsv", sep="\t")
    n_reads_used.to_csv(f"{outdir}/n_reads_used.tsv", sep="\t")

    # Score PTRs: TODO

    # Cleanup intermediate files
    for subdir in ["trimmed", "merged", "stats", "filtered", "derep"]:
        os.system(f"rm -r {outdir}/{subdir}")


if __name__ == "__main__":
    run_aptr()
