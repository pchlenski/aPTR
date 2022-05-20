import pandas as pd
import re

DD = "../data/"

primers = {
    '8F' :              'AGAGTTTGATCCTGGCTCAG',
    '27F' :             'AGAGTTTGATCMTGGCTCAG',
    'CYA106F' :         'CGGACGGGTGAGTAACGCGTGA',
    'CC [F]' : 	        'CCAGACTCCTACGGGAGGCAGC',
    '357F' :            'CTCCTACGGGAGGCAGCAG',
    'CYA359F' :         'GGGGAATYTTCCGCAATGGG',
    '515F' :            'GTGCCAGCMGCCGCGGTAA',
    '533F' :            'GTGCCAGCAGCCGCGGTAA',
    '895F' :            'CRCCTGGGGAGTRCRG',
    '16S.1100.F16' :    'CAACGAGCGCAACCCT',
    '1237F' :           'GGGCTACACACGYGCWAC',
    '519R' :            'GWATTACCGCGGCKGCTG',
    'CYA781R' :         'GACTACWGGGGTATCTAATCCCWTT',
    'CD [R]' : 	        'CTTGTGCGGGCCCCCGTCAATTC',
    '902R' :            'GTCAATTCITTTGAGTTTYARYC',
    '904R' :            'CCCCGTCAATTCITTTGAGTTTYAR',
    '907R' :            'CCGTCAATTCMTTTRAGTTT',
    '1100R' :           'AGGGTTGCGCTCGTTG',
    '1185mR' :          'GAYTTGACGTCATCCM',
    '1185aR' :          'GAYTTGACGTCATCCA',
    '1381R' :           'CGGTGTGTACAAGRCCYGRGA',
    '1381bR' :          'CGGGCGGTGTGTACAAGRCCYGRGA',
    '1391R' :           'GACGGGCGGTGTGTRCA',
    '1492R (l)' :       'GGTTACCTTGTTACGACTT',
    '1492R (s)' :	    'ACCTTGTTACGACTT',
    '926R' :            'CCGYCAATTYMTTTRAGTTT', # right for SRS8281217
    '515FB' :           'GTGYCAGCMGCCGCGGTAA' # left for SRS8281217
}



key = {
    'r': 'a|g',
    'y': 'c|t',
    'm': 'a|c',
    'w': 'a|t',
    'k': 'g|t',
    'a': 'a',
    'c': 'c',
    'g': 'g',
    't': 't'
}



def find_primer(seq, primer):
    # For cleanliness
    if primer is None:
        return seq
    
    # Turn primer into regex
    re_primer = ''.join([key[x] for x in primer.lower()])
    pattern = re.compile(re_primer)
    out = pattern.split(seq, maxsplit=1)
    return out



def trim_primers(seq, left, right):
    # Input stuff
    seq = str(seq).lower()

    # Left side
    if left is not None and left != "":
        trim_left = find_primer(seq, left)
        seq = trim_left[-1]

    # Right side
    if right is not None and right != "":
        trim_right = find_primer(seq, right)
        seq = trim_right[0]

    # Output - this only happens if right and left adapters are found
    return seq



def filter_db(
    path_to_dnaA=f"{DD}/allDnaA.tsv",
    path_to_16s=f"{DD}/allSSU.tsv",
    left_primer=None,
    right_primer=None) -> pd.DataFrame:
    """ Filter DB by adapters, return candidate sequences """

    # Get tables
    dnaA_table = pd.read_table(path_to_dnaA, dtype={0:str})
    ssu_table = pd.read_table(path_to_16s, dtype={0:str}) # 'dtype' sets genome ID as string

    # Clean up by DnaA:
    dnaA_table = dnaA_table[dnaA_table['feature.product'] == 'Chromosomal replication initiator protein DnaA']
    dnaA_table = dnaA_table.drop_duplicates('genome.genome_id')

    # Merge tables
    table = pd.merge(
        ssu_table, dnaA_table,
        how='inner',
        on=['genome.genome_id', 'feature.accession'], 
        suffixes=['_16s', '_dnaA']
    )

    # Add 16S substring
    table['filtered_seq'] = [trim_primers(x, left_primer, right_primer) for x in table['feature.na_sequence']]

    # Iteratively filter on sequence
    diff = 1
    bad_seqs = set()
    while diff > 0:
        # Find contigs with a single sequence
        table_by_contigs = table.groupby('feature.accession').nunique()
        bad_contigs_idx = table_by_contigs['filtered_seq'] == 1
        bad_contigs = table_by_contigs[bad_contigs_idx]['filtered_seq'].index

        # All sequences appearing in a bad contig are bad sequences
        bad_seqs = bad_seqs | set(table[table['feature.accession'].isin(bad_contigs)]['filtered_seq'])

        # Throw out any appearances of bad sequences
        table_filtered = table[~table['filtered_seq'].isin(bad_seqs)]
        diff = len(table) - len(table_filtered)
        table = table_filtered

    # Clean up and return table
    table = table[[
        'genome.genome_id',
        'genome.contigs_16s',
        'feature.accession',
        'feature.patric_id_16s',
        'feature.start_16s',
        'feature.start_dnaA',
        'genome.genome_length_16s',
        'filtered_seq'
    ]]
    table.columns = [
        'genome',
        'n_contigs',
        'contig',
        'feature',
        '16s_position',
        'oor_position',
        'size',
        '16s_sequence'
    ]

    return table

def generate_vsearch_db(db, output_file=f"{DD}vsearch_db.fa"):
    with open(output_file, "w+") as f:
        for i,(id,seq) in db[["feature", "16s_sequence"]].iterrows():
            print(f">{id}\n{seq}", file=f)