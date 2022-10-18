""" Database operations """

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple

from src.oor_distance import oor_distance
from .new_filter import filter_db
import warnings


class RnaDB:
    def __init__(self, load: str = False, **kwargs):
        """
        Initialize a DB instance with an appropriately-trimmed DB.
        Passes kwargs to new_filter.filter_db().
        """

        # Suppress SettingWithCopyWarning, which is not our fault here:
        pd.set_option("mode.chained_assignment", None)

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
        self.complete_genomes = self.db[self.db["n_contigs"] == 1][
            "genome"
        ].unique()
        self.md5s = self.db["md5"].unique()

    def find_genomes_by_md5(self, md5s: List[str]) -> List[str]:
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
        return list(self.db[self.db["md5"].isin(md5s)]["genome"].unique())

    def generate_genome_objects(
        self, genome_ids: list
    ) -> Tuple[List[Dict[str, List[int]]], List[str]]:
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
        if not isinstance(genome_ids, list):
            genome_ids = [genome_ids]

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
                [
                    oor_distance(position=pos, oor=oor, size=size)
                    for pos in rna_positions
                ]
            )

            # Maintain consistent indexing scheme for sequence MD5s
            seqs = []
            for seq in genome_data["md5"]:
                if seq not in all_seqs:
                    all_seqs.append(seq)
                seqs.append(all_seqs.index(seq))

            # Output dict
            out.append({"id": genome_id, "pos": dist, "seqs": seqs})

        # Generate genome to seq matrix
        seqs = out[0]["seqs"]
        m = len(seqs)
        k = np.max(seqs) + 1
        g2s = np.array([np.eye(k)[seq] for seq in seqs])

        return out, all_seqs, g2s

    def __getitem__(self, key: str) -> pd.DataFrame:
        """Return a subset of the DB corresponding to a genome ID"""
        if key in self.genomes:
            return self.db[self.db["genome"] == key]
        elif key in self.md5s:
            return self.db[self.db["md5"] == key]
        else:
            raise ValueError(f"No match in DB for {key}")
