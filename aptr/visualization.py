import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def seq_sharing(genome_ids, db):
    """Plot a network where nodes are genomes and edges are shared sequences"""
    seqs = [set(db[genome_id]["md5"]) for genome_id in genome_ids]
    adj = np.zeros((len(genome_ids), len(genome_ids)))
    for i, seq_i in enumerate(seqs):
        for j, seq_j in enumerate(seqs):
            adj[i, j] = len(seq_i & seq_j)
    G = nx.from_numpy_matrix(adj)
    G.labels = {i: genome_id for i, genome_id in enumerate(genome_ids)}
    nx.draw(G, labels=G.labels)
    plt.show()
