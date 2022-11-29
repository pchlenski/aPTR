# IUPAC codes for nucleotides
# https://www.bioinformatics.org/sms/iupac.html
key = {
    "n": "(?:a|c|g|t)",
    "x": "(?:a|c|g|t)",
    "d": "(?:a|g|t)",
    "h": "(?:a|c|t)",
    "v": "(?:a|c|g)",
    "b": "(?:c|g|t)",
    "r": "(?:a|g)",
    "y": "(?:c|t)",
    "m": "(?:a|c)",
    "w": "(?:a|t)",
    "k": "(?:g|t)",
    "s": "(?:c|g)",
    "a": "a",
    "c": "c",
    "g": "g",
    "t": "t",
}

# Shorthand for primers
primers = {
    "8F": "AGAGTTTGATCCTGGCTCAG",
    "27F": "AGAGTTTGATCMTGGCTCAG",
    "CYA106F": "CGGACGGGTGAGTAACGCGTGA",
    "CC [F]": "CCAGACTCCTACGGGAGGCAGC",
    "357F": "CTCCTACGGGAGGCAGCAG",
    "CYA359F": "GGGGAATYTTCCGCAATGGG",
    "515F": "GTGCCAGCMGCCGCGGTAA",
    "533F": "GTGCCAGCAGCCGCGGTAA",
    "895F": "CRCCTGGGGAGTRCRG",
    "16S.1100.F16": "CAACGAGCGCAACCCT",
    "1237F": "GGGCTACACACGYGCWAC",
    "519R": "GWATTACCGCGGCKGCTG",
    "CYA781R": "GACTACWGGGGTATCTAATCCCWTT",
    "CD [R]": "CTTGTGCGGGCCCCCGTCAATTC",
    "902R": "GTCAATTCITTTGAGTTTYARYC",
    "904R": "CCCCGTCAATTCITTTGAGTTTYAR",
    "907R": "CCGTCAATTCMTTTRAGTTT",
    "1100R": "AGGGTTGCGCTCGTTG",
    "1185mR": "GAYTTGACGTCATCCM",
    "1185aR": "GAYTTGACGTCATCCA",
    "1381R": "CGGTGTGTACAAGRCCYGRGA",
    "1381bR": "CGGGCGGTGTGTACAAGRCCYGRGA",
    "1391R": "GACGGGCGGTGTGTRCA",
    "1492R (l)": "GGTTACCTTGTTACGACTT",
    "1492R (s)": "ACCTTGTTACGACTT",
    "926R": "CCGYCAATTYMTTTRAGTTT",  # right for SRS8281217
    "515FB": "GTGYCAGCMGCCGCGGTAA",  # left for SRS8281217
    "Illumina1": "TCGTCGGCAGCGTCAGATGTGTATAAGAGACAGCCTACGGGNGGCWGCAG",  # left for Italian study
    "Illumina2": "GTCTCGTGGGCTCGGAGATGTGTATAAGAGACAGGACTACHVGGGTATCTAATCC",  # right for Italian study
    "806R": "GGACTACHVGGGTWTCTAAT",
}


def rc(seq):
    """Reverse complement of a DNA sequence. Assumes lowercase"""
    seq = seq.lower()
    # return seq.translate(str.maketrans("acgt", "tgca"))[::-1]
    return seq.translate(str.maketrans("nxdhvbrymwksacgt", "nxhdbvyrkwmstgca"))[
        ::-1
    ]  # Allow ambiguous bases
