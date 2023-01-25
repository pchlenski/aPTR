""" Database operations """

import pandas as pd
import numpy as np
from typing import Iterable, List, Dict, Tuple

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
        self.genomes = list(self.db["genome"].unique())
        self.complete_genomes = list(
            self.db[self.db["n_contigs"] == 1]["genome"].unique()
        )
        self.md5s = list(self.db["md5"].unique())

    def find_genomes_by_md5(self, md5s: List[str], strict: bool = False) -> List[str]:
        """
        Given an md5-hashed seq, return all genome IDs with that sequence

        Args:
        -----
        md5: str
            md5-hashed sequence to search for.
        strict: bool
            If True, only return genomes for which EVERY md5 sequence is
            present in the query. Otherwise, return genomes for which ANY
            md5 sequence is present in the query.

        Returns:
        --------
        list:
            All genomes IDs matching the given md5 hash.
        """
        candidates = list(self[md5s]["genome"].unique())
        if strict:
            out = []
            for genome in candidates:
                genome_md5s = self[genome]["md5"].unique()
                if np.all([md5 in md5s for md5 in genome_md5s]):
                    out.append(genome)
            return out
        else:
            return candidates

    def generate_genome_objects(
        self, genome_ids: list, from_md5s: bool = False
    ) -> Tuple[List[Dict[str, List[int]]], List[str]]:
        """
        Given a genome ID, return a 'genome' object

        Args:
        -----
        genome_ids: list
            List of genome IDs to generate genome objects for.
        from_md5s: bool
            If True, interpret genome_ids as md5 hashes instead of genome IDs.

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

        if isinstance(genome_ids, str):
            genome_ids = [genome_ids]

        out = []
        all_seqs = []

        if from_md5s:
            genomes = self.find_genomes_by_md5(genome_ids)

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
            # TODO: try to do this with np.apply_along_axis

            # Maintain consistent indexing scheme for sequence MD5s
            seqs = []
            for seq in genome_data["md5"]:
                if seq not in all_seqs:
                    all_seqs.append(seq)
                seqs.append(all_seqs.index(seq))

            # Output dict
            out.append({"id": genome_id, "pos": dist, "seqs": seqs})

        # Generate genome-to-seq matrix
        k = len(all_seqs)
        g2s = np.array([np.eye(k)[seq] for g in out for seq in g["seqs"]])
        # This horrible list comprension does the following:
        # For each genome, create a one-hot vector showing how that sequence
        # maps to the overall OTU table, then stacks them together.

        return out, all_seqs, g2s

    def __getitem__(self, key: str) -> pd.DataFrame:
        """Return a subset of the DB corresponding to a genome ID"""
        if isinstance(key, str):
            if key in self.genomes:
                return self.db[self.db["genome"] == key]
            elif key in self.md5s:
                return self.db[self.db["md5"] == key]
        else:
            key = list(key)
            if np.any([x in self.genomes for x in key]):
                return self.db[self.db["genome"].isin(key)]
            elif np.any([x in self.md5s for x in key]):
                return self.db[self.db["md5"].isin(key)]

        raise ValueError(f"No match in DB for {key}")
