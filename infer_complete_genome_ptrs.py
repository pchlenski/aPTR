"""
Compute PTR estimates for the "complete_1e*" directories
"""

import pandas as pd
from src.solver import *

db = pd.read_pickle("./data/db.pkl")
otus = pd.read_table("./out/eumaeus/complete_1e5/16s_otus.tsv")

print(otus)