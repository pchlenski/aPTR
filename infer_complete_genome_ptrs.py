"""
Compute PTR estimates for the "complete_1e*" directories
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.solver import *

my_db = pd.read_pickle("./data/db.pkl")
my_collisions = pd.read_pickle("./data/collisions.pkl")
my_otus = pd.read_table("./out/6bc6c418-9e20-4b2b-93a5-603a912a128d/16s_otus.tsv", dtype={0: str})
my_otus = my_otus.set_index('otu')

my_true_ptrs = pd.read_table("./out/6bc6c418-9e20-4b2b-93a5-603a912a128d/ptrs.tsv", dtype={0: str})
my_true_ptrs = my_true_ptrs.set_index('genome')

# Solve PTRs as provided in 'otus.tsv':
ptrs = solve_matrix(my_db, my_otus, my_collisions, my_true_ptrs)
plt.matshow(ptrs.pivot('genome', 'sample', 'err'))
plt.colorbar()
plt.show()
