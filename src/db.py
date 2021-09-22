"""
Implements the 16S database class
"""

import pandas as pd

class RnaDB():
    """
    A database of diverged 16S RNA sequences and genomes, plus utility functions.

    Attributes:
    -----------
    db:
        A database of 16S sequences and locations
    collisions:
        A database of 16S sequences which collide with DB
    """

    def __init__(
        self,
        db_path : str,
        collisions_path : str) -> None:
        """
        Initialize DB: load databases

        Args:
        -----
        db_path:
            Path to 16S database.
        collisions:
            Path to collisions database.

        Returns:
        --------
        An RnaDB object loaded with 16S and collision databases loaded.

        Raises:
        -------
        None.
        """
        self.db = pd.read_pickle(db_path)
        self.collisions = pd.read_pickle(collisions_path)

        self.genomes = list(self.db["genome"].unique())
        self.md5s = list(self.db["16s_md5"].unique())

    def __getitem__(self, gid : str) -> pd.DataFrame:
        """
        Get a subdatabase by genome ID

        Args:
        -----
        gid:
            String or list of strings. A genome ID or contig ID.

        Returns:
        --------
        A Pandas dataframe filtered by genome ID.

        Raises:
        --------
        TypeError:
            If 'gid' argument is not a string.
        ValueError:
            If 'gid' argument does not match any genomes or contigs
        """

        if isinstance(gid, str):
            gid = [gid]
        elif isinstance(gid, list):
            for element in gid:
                if not isinstance(element, str):
                    raise TypeError(f"ID list contains non-string element {element}")
        else:
            raise TypeError(f"ID should be str, not {type(gid)}")

        # TODO: check collisions too

        if self.db["genome"].isin(gid).any():
            return self.db[self.db["genome"].isin(gid)]
        elif self.db["contig"].isin(gid).any():
            return self.db[self.db["contig"].isin(gid)]
        else:
            raise ValueError(f"No contig or genome matches for '{gid}'")

    def md5_to_genomes(self, md5 : str) -> (list, list):
        """
        Given an OTU's 16S md5 hash, return all genomes in which it is contained

        Args:
        -----
        md5:
            String. The md5 hash of a given 16S sequence.

        Returns:
        --------
        db_matches:
            A list of genome IDs matching this OTU in 'database.'
        db_collisions:
            A list of genome IDs matching this OTU in 'collisions.'

        Raises:
        -------
        None.
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
        db_matches:
            A list of md5s matching this genome ID in 'database.'
        db_collisions:
            A list of md5s matching this genome ID in 'collisions.'

        Raises:
        -------
        None.
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

