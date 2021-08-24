"""
This is a utility script to filter the data/seqs directory down to just complete genomes.
"""

import os
import pandas as pd

# Load everything we need
path = "./data/seqs"
files = os.listdir(path)
db = pd.read_pickle("./data/db.pkl")
complete_genomes = db[db['n_contigs'] == 1]['genome'].unique()

# Make new dir
try:
    os.mkdir("./data/seqs_complete")
except:
    pass

# Filter and copy
for file in files:
    prefix = file.split(".fna.gz")[0]
    if prefix in complete_genomes:
        os.system(f"cp ./data/seqs/{file} ./data/seqs_complete/{file}")
