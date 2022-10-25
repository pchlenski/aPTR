#!/usr/bin/env python3

""" This is the command-line tool that runs the entire aPTR pipeline """

import os
import uuid
import argparse
import pandas as pd
import pickle
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
        "--otu_path",
        type=str,
        help="Path to an OTU table. Skips preprocessing.",
    )
    return parser.parse_args()


def run_aptr():
    """Run the aPTR pipeline"""
    # Get arguments
    args = get_args()

    # Suppress pandas indexing warning
    pd.options.mode.chained_assignment = None

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

    # All the preprocessing takes place here
    if args.otu_path:
        otu_path = args.otu_path
    else:
        preprocess_samples(
            path=args.path,
            adapter1=args.adapter1,
            adapter2=args.adapter2,
            db_path=db_fasta_path,
            outdir=outdir,
        )
        otu_path = f"{outdir}/filtered/otu_table.tsv"

    # Infer PTRs
    otus = pd.read_table(otu_path, index_col=0)
    solver = TorchSolver(md5s=otus.index, otus=otus, db=db)

    solver.train(lr=0.1, tolerance=1e-6)
    inferred_ptrs = pd.DataFrame(
        data=solver.B_hat.exp().detach().numpy(),
        index=solver.genome_ids,
        columns=solver.sample_ids,
    )
    inferred_abundances = pd.DataFrame(
        data=solver.A_hat.exp().detach().numpy(),
        index=solver.genome_ids,
        columns=solver.sample_ids,
    )

    # Save inferred quantities
    inferred_ptrs.to_csv(f"{outdir}/inferred_ptrs.tsv", sep="\t")
    inferred_abundances.to_csv(f"{outdir}/inferred_abundances.tsv", sep="\t")

    # Score PTRs: TODO

    # Cleanup intermediate files
    for subdir in ["trimmed", "merged", "stats", "filtered", "derep"]:
        os.system(f"rm -r {outdir}/{subdir}")


if __name__ == "__main__":
    run_aptr()
