scores = []
for scale_exp in range(6):
    i = 10 ** scale_exp
    print(f"Scale: {i}")
    samples, ptrs, coverages, otus = simulate_from_ids(
        db=db,
        ids=sample_genomes,
        fasta_path="/home/phil/aptr/data/seqs",
        n_samples=1,
        scale=i,
        shuffle=False,  # Suppress shuffling to conserve memory
        skip_wgs=True,
        verbose=False, # Shut up
    )
    score = score_simulation(
        otus=otus,
        ptrs=ptrs,
        otu_table_path="/tmp/otus.tsv",
        db_path="/home/phil/aptr/experiments/simulated_complete/1e5/aptr_out/db.pkl",
        torch=True,
        n_epochs=100000
    )
    scores.append(score)
    print(f"Score: {score}")
    print()
