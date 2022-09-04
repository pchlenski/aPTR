""" Database operations """

import pandas as pd
import numpy as np
from src.new_filter import _filter_db


class RnaDB:
    def __init__(self, load=False, left_primer=None, right_primer=None):
        """ Initialize a DB instance with an appropriately-trimmed DB"""
        if load:
            self.db = pd.read_pickle(load)
        else:
            self.db = _filter_db(left_primer=left_primer, right_primer=right_primer)
            self.left_primer = left_primer
            self.right_primer = right_primer
        self.genomes = self.db["genome"].unique()
        self.md5s = self.db["md5"].unique()

    def find_genomes_by_md5(self, md5):
        """ Given an md5-hashed seq, return all genome IDs with that sequence """
        return self.db[self.db["md5"] == md5]["genome"].unique()

    def get_oor_dist(self, oor, pos, length):
        """ Given an OOR coordinate, a position, and a length, return distance """
        oor = float(oor)
        pos = float(pos)
        length = float(length)
        d1 = np.abs(oor - pos)
        d2 = np.abs(oor + length - pos)
        return np.minimum(d1, d2) / length

    def generate_genome_objects(self, genome_ids):
        """ Given a genome ID, return a 'genome' object """
        out = []
        all_seqs = []
        for genome_id in genome_ids:
            genome_data = self[genome_id]
            oor = genome_data.iloc[0]["oor_position"]
            size = genome_data.iloc[0]["size"]

            # Throw away unknown OOR
            if np.isnan(oor):
                print("OOR is not specified")
                raise ValueError()

            # Get distances
            rna_positions = np.array(genome_data["16s_position"], dtype=float)
            dist = np.array(
                [self.get_oor_dist(oor, pos, size) for pos in rna_positions]
            )

            # Maintain consistent indexing scheme for sequence MD5s
            seqs = []
            for seq in genome_data["md5"]:
                if seq not in all_seqs:
                    all_seqs.append(seq)
                seqs.append(all_seqs.index(seq))

            # Output dict
            out.append({"id": genome_id, "pos": dist, "seqs": seqs})

        return out, all_seqs

    def __getitem__(self, key):
        if key in self.genomes:
            return self.db[self.db["genome"] == key]
        elif key in self.md5s:
            return self.db[self.db["md5"] == key]
        else:
            raise ValueError(f"No match in DB for {key}")
