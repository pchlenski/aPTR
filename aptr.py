#!/usr/bin/env python3

import sys
import os
from src.process_samples import process_samples
from src.matrix_solver import OTUSolver
from src.new_filter import filter_db, generate_vsearch_db

# Need to have at least path and adapter sequences
if len(sys.argv) < 4:
    raise ValueError("Not enough values!")

# Case when database is not provided in advance
elif len(sys.argv) == 4:
    _, path, adapter1, adapter2 = sys.argv

    if adapter1 == "":
        adapter1 = None
    if adapter2 == "":
        adapter2 = None

    db = filter_db(
        path_to_dnaA = "./data/allDnaA.tsv",
        path_to_16s = "./data/allSSU.tsv",
        left_primer = adapter1,
        right_primer = adapter2
    )
    db_path = f"{path}/aPTR_out/db.fasta"
    try:
        os.mkdir(f"{path}/aPTR_out")
    except FileExistsError:
        pass

    generate_vsearch_db(db, output_file=db_path)

# Case when database path is also given
else:
    _, path, adapter1, adapter2, db_path = sys.argv

# All the action takes place here
process_samples(path, adapter1, adapter2, db_path=db_path)