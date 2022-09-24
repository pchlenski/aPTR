""" Database operations """

import pandas as pd
import numpy as np
from typing import List, Dict
from .new_filter import filter_db


class RnaDB:
    def __init__(self, load: str = False, **kwargs):
        """
        Initialize a DB instance with an appropriately-trimmed DB.
        Passes kwargs to new_filter.filter_db().
        """

        # Get the DB
        if load:
            self.db = pd.read_pickle(load)
        else:
            self.db = filter_db(**kwargs)
            if "left_primer" in kwargs:
                self.left_primer = kwargs["left_primer"]
            else:
                self.left_primer = None

            if "right_primer" in kwargs:
                self.right_primer = kwargs["right_primer"]
            else:
                self.right_primer = None

        # Set some other useful attributes
        self.genomes = self.db["genome"].unique()
        self.md5s = self.db["md5"].unique()

    def find_genomes_by_md5(self, md5: str) -> List[str]:
        """
        Given an md5-hashed seq, return all genome IDs with that sequence

        Args:
        -----
        md5: str
            md5-hashed sequence to search for.

        Returns:
        --------
        list:
            All genomes IDs matching the given md5 hash.
        """
        return list(self.db[self.db["md5"] == md5]["genome"].unique())

    def get_oor_dist(self, oor: int, pos: int, length: int) -> float:
        """
        Given an OOR coordinate, a position, and a length, return distance.

        Args:
        -----
        oor: int
            Locus at which origin of replication is located.
        pos: int
            Position for which coverage is being computed.
        length: int
            Length of the genome.

        Returns:
        --------
        float:
            Distance from the OOR to the position, normalized to be in [0, 0.5].

        Raises:
        -------
        ValueError: if the OOR is not in [0, length]
        ValueError: if the position is not in [0, length]
        ValueError: if the length is negative
        """

        # Input validation
        oor = float(oor)
        pos = float(pos)
        length = float(length)
        if oor < 0 or oor > length:
            raise ValueError("OOR must be in [0, length]")
        if pos < 0 or pos > length:
            raise ValueError("Position must be in [0, length]")
        if length < 0:
            raise ValueError("Length must be positive")

        # Compute distance
        d1 = np.abs(oor - pos)
        d2 = np.abs(oor + length - pos)
        return np.minimum(d1, d2) / length

    def generate_genome_objects(
        self, genome_ids: list
    ) -> List[List[Dict[str, List[int]]], List[str]]:
        """
        Given a genome ID, return a 'genome' object

        Args:
        -----
        genome_ids: list
            List of genome IDs to generate genome objects for.

        Returns:
        --------
        list:
            List of genome objects corresponding to the input genome IDs.
        list:
            List of all sequences/md5s used

        Raises:
        -------
        ValueError: if any genome ID is not in the DB
        """
        out = []
        all_seqs = []
        for genome_id in genome_ids:
            if genome_id not in self.genomes:
                raise ValueError(f"No match in DB for {genome_id}")
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

    def __getitem__(self, key: str) -> pd.DataFrame:
        """Return a subset of the DB corresponding to a genome ID"""
        if key in self.genomes:
            return self.db[self.db["genome"] == key]
        elif key in self.md5s:
            return self.db[self.db["md5"] == key]
        else:
            raise ValueError(f"No match in DB for {key}")
