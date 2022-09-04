#!/usr/bin/env python3

""" This is the command-line tool that runs the entire aPTR pipeline """

import sys
import os
import uuid
from src.process_samples import process_samples
from src.new_filter import filter_db, generate_vsearch_db

# Need to have at least path and adapter sequences
if len(sys.argv) < 4:
    raise ValueError("Not enough values!")

# Case when database is not provided in advance
elif len(sys.argv) == 4:
    _, path, adapter1, adapter2 = sys.argv
    outdir = f"{path}/aptr_{uuid.uuid4()}"

    db = filter_db(
        path_to_dnaA="./data/allDnaA.tsv",
        path_to_16s="./data/allSSU.tsv",
        left_primer=adapter1,
        right_primer=adapter2,
    )
    db_path = f"{outdir}/db.fasta"
    try:
        os.mkdir(outdir)
    except FileExistsError:
        pass

    # Save a reduced database with adapters cut
    generate_vsearch_db(db, output_file=db_path)
    db.to_pickle(f"{outdir}/db.pkl")

# Case when database path is also given
else:
    _, path, adapter1, adapter2, db_path = sys.argv

# All the action takes place here
process_samples(
    path=path, adapter1=adapter1, adapter2=adapter2, db_path=db_path, outdir=outdir
)
