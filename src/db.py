"""
Implements the 16S database class
"""

import pandas as pd

class RnaDB():
    """
    TODO
    """

    def __init__(self,
        db_path : str,
        collisions_path : str) -> None:
        """
        Initialize DB: load databases
        """
        self.db = pd.read_pickle(db_path)
        self.collisions = pd.read_pickle(collisions_path)

    def md5_to_genomes(self, md5 : str) -> (list, list):
        """
        Given an OTU's 16S md5 hash, return all genomes in which it is contained

        Args:
        -----
        md5:
            String. The md5 hash of a given 16S sequence.

        Returns:
        --------
        The following two lists:
        * db_matches: a list of genome IDs matching this OTU in 'database'
        * db_collisions: a list of genome IDs matching this OTU in 'collisions'

        Raises:
        -------
        TODO
        """

        # Check DB
        db_matches = self.db[self.db['16s_md5'] == md5]['genome'].unique()

        # Check collisions
        db_collisions = self.collisions[self.collisions['16s_md5'] == md5]['genome'].unique()

        return list(db_matches), list(db_collisions)

    def genome_to_md5s(self, gid : str) -> (list, list):
        """
        Given a genome ID, return all md5s it contains

        Args:
        -----
        gid:
            String. The genome ID of a given organism.

        Returns:
        --------
        The following two lists:
        * db_matches: a list of md5s matching this genome ID in 'database'
        * db_collisions: a list of md5s matching this genome ID in 'collisions'

        Raises:
        -------
        TODO
        """

        # Check DB
        db_matches = self.db[self.db['genome'] == gid]['16s_md5'].unique()

        # Check collisions
        db_collisions = self.collisions[self.collisions['genome'] == gid]['16s_md5'].unique()

        return list(db_matches), list(db_collisions)

    def match_otus(self, fasta_path : str) -> dict:
        """
        Maps a fasta file of 16S sequences against the database.
        TODO
        """

        raise NotImplementedError

