"""
Implements the 16S database class
"""

import numpy as np
import pandas as pd
from .solver import solver

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

    def solve_genome(
        self,
        genome_id : str,
        sample_id : str,
        otus : pd.DataFrame,
        true_ptrs : pd.DataFrame,
        regularization : float = 0) -> dict:
        """
        Given a genome ID, a DB, and some coverages, estimate the PTR.

        Args:
        -----
        genome_id:
            String. Genome ID as provided by md5_to_genomes().
        sample_id:
            Integer or string. Sample number for the given genome.
        otus:
            Pandas DataFrame. A matrix of 16S OTU read/abundance counts.
        true_ptrs:
            Pandas DataFrame. A matrix of true PTR values, if known.
        regularization:
            Float. L2-regularization penalty applied to m and b.

        Returns:
        --------
        A dict with keys 'genome', 'sample', 'ptr', 'true_ptr' to be appended to a results DB.

        Raises:
        -------
        TODO
        """
        # Build up x_values
        db_matched = self[genome_id]
        x_positions = db_matched['16s_position'] / db_matched['size']

        # Build up mappings
        md5s_matched = db_matched['16s_md5']
        mapping = {}
        idx = 0
        # TODO: vectorize this
        for md5 in md5s_matched:
            if md5 not in mapping:
                mapping[md5] = idx
                idx += 1
        x_mapping = [mapping[x] for x in md5s_matched]

        # Sort md5s by their index
        md5s = [None for _ in mapping]
        for md5 in mapping:
            md5s[mapping[md5]] = md5

        # Build up coverages
        coverages = otus[sample_id].reindex(md5s)

        # Send to solver
        results = solver(
            x_values=x_positions, 
            mappings=x_mapping, 
            coverages=coverages, 
            regularization=regularization
        )

        # Append output to PTRs dataframe
        m = results[0]
        b = results[1]
        peak = np.exp2(b)
        trough = np.exp2(m * 0.5 + b)
        ptr = peak / trough

        # Get true PTR
        if true_ptrs is not None:
            try:
                true_ptr = true_ptrs.loc[genome_id, sample_id]
            except KeyError as e:
                true_ptr = np.nan
                print(f"True PTR evaluation: Bypassing KeyError: {e}")
        else:
            true_ptr = None

        return {"genome" : genome_id, "sample" : sample_id, "ptr" : ptr, "true_ptr" : true_ptr}

    def solve_sample(
        self,
        sample_id : str,
        otus : pd.DataFrame,
        true_ptrs : pd.DataFrame,
        regularization : float = 0) -> list:
        """
        Given a sample name, solve all available 16S systems.

        Args:
        -----
        sample_id:
            String. The name of a given sample. Used for indexing into OTU matrix.
        otus:
            Pandas DataFrame. A matrix of 16S OTU read/abundance counts.
        true_ptrs:
            Pandas DataFrame. A matrix of true PTR values, if known.
        regularization:
            Float. L2-regularization penalty applied to m and b.

        Returns:
        --------
        A list of dicts (solve_genome outputs) to be appended to an output DB.

        Raises:
        -------
        TODO
        """
        genomes = []
        out = []
        # Build up lists of genomes and md5s
        for md5 in otus.index:
            if otus.loc[md5, sample_id] > 0:
                match, coll = self.md5_to_genomes(md5)

                # Skip collisions for now... these are overdetermined
                if coll:
                    print(f"Collision for {md5}. Skipping this sequence...")
                else:
                    genomes += match

        for genome_id in set(genomes):
            result = self.solve_genome(
                genome_id, sample_id, 
                otus=otus, 
                true_ptrs=true_ptrs, 
                regularization=regularization
            )
            out.append(result)

        return out

    def solve_matrix(
        otus : pd.DataFrame,
        true_ptrs : pd.DataFrame = None,
        max_error : float = np.inf,
        regularization : float = 0) -> pd.DataFrame:
        """
        Given a 16S db, OTU read/abundance matrix, and true PTR values (optional), esimate PTRs.

        Args:
        -----
        otus:
            Pandas DataFrame. A matrix of 16S OTU read/abundance counts.
        true_ptrs:
            Pandas DataFrame. A matrix of true PTR values, if known.
        max_error:
            Float. Used to trim extreme error values (experimental).
        regularization:
            Float. L2-regularization penalty applied to m and b.

        Returns:
        --------
        A Pandas Dataframe with the following columns:
        * genome: genome ID (taken from database)
        * sample: sample ID (as given in OTU matrix)
        * ptr: PTR (estimated by tool)
        * true_ptr: PTR (provided in true_ptrs DataFrame)
        * err: absolute error (capped at 5) between true and estimated PTR

        Raises:
        -------
        TODO
        """
        out = pd.DataFrame(columns=["genome", "sample", "ptr", "true_ptr"])

        # For each column, build up x_values, mappings, coverages; send to solver
        for column in otus.columns:
            results = self.solve_sample(
                column, 
                otus=otus, 
                true_ptrs=true_ptrs, 
                regularization=regularization
            )
            out = out.append(results, ignore_index=True)

        out['err'] = np.abs(out['ptr'] - out['true_ptr'])
        # Cut off error threshold
        out['err'] = out['err'].apply(lambda x: np.min([x, max_error]))

        return out


