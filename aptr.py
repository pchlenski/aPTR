import sys
from src.process_samples import process_samples
from src.matrix_solver import OTUSolver
from src.new_filter import filter_db, generate_vsearch_db

if len(sys.argv) < 4:
    raise ValueError("Not enough values!")  

_, path, adapter1, adapter2 = sys.argv

db = filter_db(
    path_to_dnaA = "./data/allDnaA.tsv",
    path_to_16s = "./data/allSSU.tsv",
    left_primer = adapter1,
    right_primer = adapter2
)

generate_vsearch_db(db, output_file=f"{path}/db.fasta")

process_samples(path, adapter1, adapter2)