import sys
from src.process_samples import process_samples
from src.matrix_solver import OTUSolver
from src.new_filter import filter_db, generate_vsearch_db

if len(sys.argv) < 4:
    raise ValueError("Not enough values!")

elif len(sys.argv) == 4:
    _, path, adapter1, adapter2 = sys.argv
    db = filter_db(
        path_to_dnaA = "./data/allDnaA.tsv",
        path_to_16s = "./data/allSSU.tsv",
        left_primer = adapter1,
        right_primer = adapter2
    )
    db_path = f"{path}/db.fasta"
    generate_vsearch_db(db, output_file=f"{path}/db.fasta")

else:
    _, path, adapter1, adapter2, db_path = sys.argv

process_samples(path, adapter1, adapter2, db_path=db_path)