#!/usr/bin/env python3

""" This is the command-line tool that runs the entire aPTR pipeline """

import os
import uuid
import argparse
import pickle
import pandas as pd
from src.preprocess_samples import preprocess_samples
from src.new_filter import filter_db, save_as_vsearch_db
from src.solve_table import solve_all, score_predictions


def run_aptr():
    # Argument parsing
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
    args = parser.parse_args()

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
        db_path = f"{outdir}/db.fasta"
        try:
            os.mkdir(outdir)
        except FileExistsError:
            pass
        print(f"Output directory UUID: {outdir}")

        # Save a reduced database with adapters cut
        save_as_vsearch_db(db, output_file_path=db_path)
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
    inferred_ptrs, inferred_abundances = solve_all(
        otu_table_path=otu_path,
        db_path=db_path,
        left_adapter=args.adapter1,
        right_adapter=args.adapter2,
        true_values=None,  # TODO: find some true values to try on, e.g. from simulation or coPTR
    )

    inferred_ptrs.to_csv(f"{outdir}/inferred_ptrs.tsv", sep="\t")
    inferred_abundances.to_csv(f"{outdir}/inferred_abundances.tsv", sep="\t")

    # Score predictions
    ptr_scores = score_predictions(
        inferred_ptrs=inferred_ptrs,
        true_ptrs=pd.read_csv(f"{args.path}/ptrs.csv"),
    )
    print(ptr_scores, file=open(f"{outdir}/ptr_scores.txt", "w"))

    # Cleanup intermediate files
    for subdir in ["trimmed", "merged", "stats", "filtered", "derep"]:
        os.system(f"rm -r {outdir}/{subdir}")


if __name__ == "__main__":
    run_aptr()
