#!/usr/bin/env python3

""" This is the command-line tool that runs the entire aPTR pipeline """

import os
import uuid
import argparse
import pandas as pd
from src.preprocess_samples import preprocess_samples
from src.new_filter import filter_db, save_as_vsearch_db
# from src.solve_table import solve_all, score_predictions
# from src.torch_solver import solve_table
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
        help="Path to an RnaDB object. Skips database generation.",
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
        db_path = args.db_path
    else:
        db = filter_db(
            path_to_dnaA="./data/allDnaA.tsv",
            path_to_16s="./data/allSSU.tsv",
            left_primer=args.adapter1,
            right_primer=args.adapter2,
        )
        db_path = f"{outdir}/db.pkl"
        db_fasta_path = f"{outdir}/db.fasta"
        try:
            os.mkdir(outdir)
        except FileExistsError:
            pass
        print(f"Output directory UUID: {outdir}")

        # Save a reduced database with adapters cut
        save_as_vsearch_db(db, output_file_path=db_fasta_path)
        db.to_pickle(f"{outdir}/db.pkl")

    # All the preprocessing takes place here
    if args.otu_path:
        otu_path = args.otu_path
    else:
        preprocess_samples(
            path=args.path,
            adapter1=args.adapter1,
            adapter2=args.adapter2,
            db_path=db_path,
            outdir=outdir,
        )
        otu_path = f"{outdir}/filtered/otu_table.tsv"

    # Infer PTRs
    otus = pd.read_table(otu_path)
    solver = TorchSolver(
        genomes=db.generate_genome_objects()[0],
    )
    solutions = solve_table(otus=otus, db=db)
    inferred_ptrs = solutions.pivot("genome", "sample", "ptr")
    inferred_abundances = solutions.pivot("genome", "sample", "abundance")

    # Save inferred quantities
    inferred_ptrs.to_csv(f"{outdir}/inferred_ptrs.tsv", sep="\t")
    inferred_abundances.to_csv(f"{outdir}/inferred_abundances.tsv", sep="\t")

    # Score PTRs
    ptr_scores = score_predictions(
        predictions=inferred_ptrs,
        true_values=pd.read_table(f"{args.path}/ptrs.tsv", index_col=0),
    )
    print(ptr_scores, file=open(f"{outdir}/ptr_scores.txt", "w"))

    # Score abundances
    abundance_scores = score_predictions(
        predictions=inferred_abundances,
        true_values=pd.read_table(f"{args.path}/coverages.tsv", index_col=0),
    )
    print(abundance_scores, file=open(f"{outdir}/abundance_scores.txt", "w"))

    # Cleanup intermediate files
    for subdir in ["trimmed", "merged", "stats", "filtered", "derep"]:
        os.system(f"rm -r {outdir}/{subdir}")


if __name__ == "__main__":
    run_aptr()
