""" Scripts for filtering DB by PCR primers """

import pandas as pd
import numpy as np
import re
from hashlib import md5

from aptr import data_dir
from aptr.string_operations import rc, key, primers

# Data directory (global variable)
# _DD = "../data/"


# def _find_primer(seq: str, primer: str) -> list:
#     """
#     Given a sequence and a primer, returns the trimmed sequence. Handles empty
#     primers and named primers.

#     Args:
#     -----
#     seq: str
#         Nucleotide sequence to trim
#     primer: str
#         Primer with which to trim the sequence. Can be empty, a named primer, or
#         a custom primer (specified by nucleotide sequence).

#     Returns:
#     --------
#     list: [str, str] or [str]
#         If the primer is found, returns the sequence split into the sequence
#         before the first occurrence of the primer and the sequence after the
#         first occurrence of the primer. If the primer is not found, returns the
#         original sequence in a list.

#     Raises:
#     -------
#     TODO
#     """

#     # For cleanliness, handle empty primers:
#     if primer is None or primer == "":
#         return [seq]
#         # Making this a singleton array ensures that seq[-1] and seq[0] return the right value

#     # Handle named primers
#     elif primer in primers:
#         primer = primers[primer]

#     # Turn primer into regex
#     re_primer = "".join([key[x] for x in primer.lower()])
#     pattern = re.compile(re_primer)
#     out = pattern.split(seq, maxsplit=1)
#     return out


def _trim_primers(
    seq: str, left: str, right: str, reverse: bool = False, silent: bool = False
) -> str:
    """
    Trim a sequence by a left and right primer.

    Args:
    -----
    seq: str
        Nucleotide sequence to trim.
    left: str
        Primer with which to trim the sequence from the 3' end.
    right: str
        Primer with which to trim the sequence from the 5' end.
    reverse: bool
        If True, reverse-complement right primer before trimming.

    Returns:
    --------
    str: trimmed sequence

    Raises:
    -------
    TODO
    """
    # # Input stuff
    # seq = str(seq).lower()

    # # Left side
    # trim_left = _find_primer(seq, left)
    # seq = trim_left[-1]

    # # Right side
    # trim_right = _find_primer(seq, right)
    # seq = trim_right[0]

    # return seq

    # Revised: we do it this way now
    if (left is None or left == "") and (right is None or right == ""):
        return seq
    elif seq is None or seq == "" or not isinstance(seq, str) or len(seq) == 0:
        if not silent:
            print(f"Warning: sequence '{seq}' is empty or not a string")
        return ""
    else:
        # Look up named primers
        if left in primers:
            left = primers[left]
        if right in primers:
            right = primers[right]

        # The rest is a regex operation
        fwd_primer = "".join([key[x] for x in left.lower()])
        if reverse:
            right = rc(right)  # reverse complement of right primer
        rev_primer = "".join([key[x] for x in right.lower()])
        pattern = re.compile(f"({fwd_primer}.*{rev_primer})")
        match = pattern.search(seq)
        if match:
            return match.group(1)
        else:
            return ""


def filter_db(
    path_to_dnaA: str = f"{data_dir}/allDnaA.tsv",
    path_to_16s: str = f"{data_dir}/allSSU.tsv",
    left_primer: str = None,
    right_primer: str = None,
    silent: bool = False,
) -> pd.DataFrame:
    """
    Filter DB by adapters, return candidate sequences

    Args:
    -----
    path_to_dnaA: str
        Path to the DnaA gene table.
    path_to_16s: str
        Path to the 16S rRNA gene table.
    left_primer: str
        Primer with which to trim the sequence from the 3' end.
    right_primer: str
        Primer with which to trim the sequence from the 5' end.

    Returns:
    --------
    pd.DataFrame:
        Filtered table of trimmed candidate sequences. Discards any genomes that
        no longer have two or more unique candidate sequences after trimming.

    Raises:
    -------
    TODO
    """

    # Get tables
    dnaA_table = pd.read_table(path_to_dnaA, dtype={0: str})
    ssu_table = pd.read_table(
        path_to_16s, dtype={"genome.genome_id": str, "feature.na_sequence": str}
    )

    # Clean up by DnaA:
    dnaA_table = dnaA_table[
        dnaA_table["feature.product"]
        == "Chromosomal replication initiator protein DnaA"
    ]
    dnaA_table = dnaA_table.drop_duplicates("genome.genome_id")

    # Merge tables
    table = pd.merge(
        ssu_table,
        dnaA_table,
        how="inner",
        on=["genome.genome_id", "feature.accession"],
        suffixes=["_16s", "_dnaA"],
    )

    original_len = len(table)

    # Add 16S substring
    table.loc[:, "filtered_seq"] = [
        _trim_primers(x, left_primer, right_primer, silent=silent)
        for x in table["feature.na_sequence"]
    ]

    # Drop all bad values (may be redundant)
    table = table[table["filtered_seq"] != ""]
    table = table.dropna(subset=["filtered_seq"])
    table = table[table["filtered_seq"].str.len() > 0]

    if not silent:
        print(
            np.sum(table["filtered_seq"] != "") / original_len,
            "sequences remain after trimming",
        )

    # Iteratively filter on sequence
    diff = 1
    bad_seqs = set()
    while diff > 0:
        # Find contigs with a single sequence
        table_by_contigs = table.groupby("feature.accession").nunique()
        bad_contigs_idx = table_by_contigs["filtered_seq"] == 1
        bad_contigs = table_by_contigs[bad_contigs_idx]["filtered_seq"].index

        # All sequences appearing in a bad contig are bad sequences
        bad_seqs = bad_seqs | set(
            table[table["feature.accession"].isin(bad_contigs)]["filtered_seq"]
        )

        # Throw out any appearances of bad sequences
        table_filtered = table[~table["filtered_seq"].isin(bad_seqs)]
        diff = len(table) - len(table_filtered)
        table = table_filtered

    if not silent:
        print(
            np.sum(table["filtered_seq"] != "") / original_len,
            "sequences remain after filtering",
        )

    # Clean up and return table
    table = table[
        [
            "genome.genome_id",
            "genome.contigs_16s",
            "feature.accession",
            "feature.patric_id_16s",
            "feature.start_16s",
            "feature.start_dnaA",
            "genome.genome_length_16s",
            "filtered_seq",
        ]
    ]
    table.columns = [
        "genome",
        "n_contigs",
        "contig",
        "feature",
        "16s_position",
        "oor_position",
        "size",
        "16s_sequence",
    ]

    table.loc[:, "md5"] = [
        md5(str(x).encode("utf-8")).hexdigest() for x in table["16s_sequence"]
    ]

    return table


def save_as_vsearch_db(
    db: pd.DataFrame,
    output_file_path: str = f"{data_dir}/vsearch_db.fa",
    method: str = "seq",
) -> None:
    """
    Given a dataframe of candidate sequences, save in VSEARCH-compatible format.

    Args:
    -----
    db: pd.DataFrame
        Table of candidate sequences. Output by filter_db().
    output_file_path: str
        Path to which to save the VSEARCH-compatible database.
    method: str
        Method by which to name database entries. Options are:
        - 'id': Sequences named according to their genome ID.
        - 'seq': Sequences named according to their hashed sequence.

    Returns:
    --------
    None (writes to file)

    Raises:
    -------
    None
    """
    with open(output_file_path, "w+") as f:
        if method == "id":
            for _, (id, seq) in db[["feature", "16s_sequence"]].iterrows():
                print(f">{id}\n{seq}", file=f)
        elif method == "seq":
            for _, (seq, md5) in (
                db[["16s_sequence", "md5"]].drop_duplicates().iterrows()
            ):
                print(f">{md5}\n{str(seq).lower()}", file=f)
