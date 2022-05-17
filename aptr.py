import sys
from src.process_samples import process_samples
from src.matrix_solver import OTUSolver

if len(sys.argv) < 4:
    raise ValueError("Not enough values!")  

_, path, adapter1, adapter2 = sys.argv

process_samples(path, adapter1, adapter2)