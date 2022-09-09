""" 
This script will grab FASTA files for all database genomes. 
Use for simulation. 
"""

import os
from src.database import RnaDB

DD = "./data/"  # data directory

# Initialize a clean database
db_full = RnaDB(
    path_to_dnaA=f"{DD}allDnaA.tsv",
    path_to_16s=f"{DD}allSSU.tsv",
)
db = db_full.db

# Grab fasta files from PATRIC - this takes ~18m to run
ids = db["genome"].unique()
for idx, id in enumerate(ids):
    if idx % 20 == 0:
        print(f"{idx} of {db['genome'].nunique()}")
    url = f"ftp://ftp.patricbrc.org/genomes/{id}/{id}.fna"
    os.system(f"wget -qO - {url} | gzip -c > {DD}seqs/{id}.fna.gz")
