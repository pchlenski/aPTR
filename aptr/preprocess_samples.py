""" Scripts for running VSEARCH and CUTADAPT on FASTRQ reads """

import os
import sys

# Path stuff
_CUTADAPT = "cutadapt"
_VSEARCH = "vsearch"

# System variables
_N_THREADS = 4
_FASTQ_MAX_EE = 1.0
_FASTQ_MIN_LEN = 50
_FASTQ_MAX_NS = 0
_FASTQ_QMAX = 93


def _exec(
    *args, verbose: bool = False, cmdfile: str = sys.stdout, logfile: str = None, errfile: str = None, **kwargs
) -> bool:
    """
    Utility function to log and execute system calls

    Args:
    -----
    args: list
        List of arguments to pass to os.system(). Get concatenated with spaces.
    verbose: bool
        Whether to print the command to stdout or cmdfile.
    cmdfile: str
        Path to file where commands are logged.
    logfile: str
        Path to file where stdout is logged.
    errfile: str
        Path to file where stderr is logged.
    kwargs: dict
        Ignored. Absorbs superfluous arguments passed to _exec().

    Returns:
    --------
    bool
        True if the command executed successfully, False otherwise.
    """

    # Coerce to string
    cmd = " ".join([str(x) for x in args])

    if verbose:
        with open(cmdfile, "a") as f:
            print(cmd, file=f)

    if logfile is not None and ">" not in cmd:
        cmd = f"{cmd} >> {logfile}"

    if errfile is not None and "2>" not in cmd:
        cmd = f"{cmd} 2>> {errfile}"

    out = os.system(cmd)
    if out != 0:
        print(f"Failed to execute command:\t{cmd}")
        raise Exception(f"out status is {out}")

    return True


def _process_sample(
    prefix: str, suffix: str, adapter1: str, adapter2: str, in_dir: str, out_dir: str, paired: bool, **exec_args
) -> bool:
    """
    For a single (paired) read file: cut adapters, filter, dereplicate reads.

    Args:
    -----
    prefix: str
        Prefix of the FASTQ file(s). Probably an SRA run accession.
    suffix: str
        Suffix of the FASTQ file(s). Probably ".fastq.gz" or ".fq.gz".
    adapter1: str
        Adapter sequence for the 3' end of the reads. Equivalent to the CUTADAPT
        -A/-a option.
    adapter2: str
        Adapter sequence for the 5' end of the reads. Equivalent to the CUTADAPT
        -G/-g option.
    in_dir: str
        Path to directory where the FASTQ file(s) are located.
    out_dir: str
        Path to directory where the output files will be written.
    paired: bool
        Whether the reads are paired-end or not.
    exec_args: dict
        Arguments to pass to _exec().

    Returns:
    --------
    bool
        True if the command executed successfully, False otherwise.
    """

    # Shared values
    cutadapt_log = f"{out_dir}/stats/{prefix}.cutadapt.log"
    merged_reads_path = f"{out_dir}/merged/{prefix}{suffix}"
    ee_stats_path = f"{out_dir}/stats/{prefix}.stats"
    filtered_reads_path = f"{out_dir}/filtered/{prefix}.filtered.fasta"
    derep_path = f"{out_dir}/derep/{prefix}.derep.fasta"

    use_cutadapt = (adapter1 is not None and adapter2 is not None) and (adapter1 != "" and adapter2 != "")

    # For paired files: cut adapters individually, then merge pairs
    if paired:
        path1 = f"{in_dir}/{prefix}_1{suffix}"  # Reads mate pair 1
        path2 = f"{in_dir}/{prefix}_2{suffix}"  # Reads mate pair 2
        out1 = f"{out_dir}/trimmed/{prefix}_1{suffix}"  # Trimmed mate pair 1
        out2 = f"{out_dir}/trimmed/{prefix}_2{suffix}"  # Trimmed mate pair 2

        # Cutadapt part
        if use_cutadapt:
            cutadapt_args = exec_args.copy()
            _exec(
                _CUTADAPT,
                f"-A {adapter1} -a {adapter1}",
                f"-G {adapter2} -g {adapter2}",
                f"-o {out1} -p {out2}",
                f"-j {_N_THREADS}",
                f"{path1} {path2}",
                f"> {cutadapt_log}",
            )
            # Do not log with _exec() because we want a separate cutadapt log
        else:
            _exec(f"cp {path1} {out1}", **exec_args)
            _exec(f"cp {path2} {out2}", **exec_args)

        # Merge pairs
        _exec(
            f"{_VSEARCH} --fastq_mergepairs {out1}",
            f"--reverse {out2}",
            f"--threads {_N_THREADS}",
            f"--fastqout {merged_reads_path}",
            f"--fastq_eeout",
            **exec_args,
        )

    # For unpaired files, simply cut adapters
    else:
        path1 = f"{in_dir}/{prefix}{suffix}"
        out1 = f"{out_dir}/trimmed/{prefix}{suffix}"

        # Cutadapt
        if use_cutadapt:
            _exec(
                f"{_CUTADAPT}",
                f"-a {adapter1}",
                f"-g {adapter2}",
                f"-o {out1}",
                f"-j {_N_THREADS}",
                f"{path1} > {cutadapt_log}",
                **exec_args,
            )
        else:
            _exec(f"cp {path1} {out1}", **exec_args)

        # Just copy rather than merging
        _exec(f"cp {out1} {merged_reads_path}", **exec_args)

    # Quality stuff
    try:
        _exec(
            _VSEARCH,
            f"--fastq_eestats2 {merged_reads_path}",
            f"--output {ee_stats_path}",
            f"--fastq_qmax {_FASTQ_QMAX}",
            **exec_args,
        )
        _exec(
            _VSEARCH,
            f"--fastq_filter {merged_reads_path}",
            f"--fastq_maxee {_FASTQ_MAX_EE}",
            f"--fastq_minlen {_FASTQ_MIN_LEN}",
            f"--fastq_maxns {_FASTQ_MAX_NS}",
            f"--fastaout {filtered_reads_path}",
            f"--fastq_qmax {_FASTQ_QMAX}",
            f"--fasta_width 0",
            **exec_args,
        )
        _exec(
            _VSEARCH,
            f"--derep_fulllength {filtered_reads_path}",
            f"--strand plus",
            f"--sizeout",
            f"--relabel {prefix}.",
            f"--output {derep_path}",
            f"--fasta_width 0",
            **exec_args,
        )
    except Exception:
        pass

    return True


def preprocess_samples(
    path: str,
    adapter1: str,
    adapter2: str,
    db_fasta_path: str,
    readcounts_path: str,
    cutoff: float,
    outdir: str,
    verbose: bool = True,
    log: bool = True,
) -> bool:
    """
    Process all samples in a directory. Generates an OTU abundance table.

    Args:
    -----
    path: str
        Path to parent of "reads" directory containing FASTQ files.
    adapter1: str
        Adapter sequence for the 3' end of the reads. Equivalent to the CUTADAPT
        -A/-a option.
    adapter2: str
        Adapter sequence for the 5' end of the reads. Equivalent to the CUTADAPT
        -G/-g option.
    db_fasta_path: str
        Path to a FASTA file of 16S sequences to search against.
    readcounts_path: str
        Path to a FASTA file of read counts (generated by VSEARCH).
    outdir: str
        Path to directory where the output files will be written.
    verbose: bool
        Whether to print the output of the commands to the console.
    log: bool
        Whether to log the output of the commands to a file.

    Returns:
    --------
    bool
        True if the command executed successfully, False otherwise.
    """

    # Ensure all directories exist
    reads_dir = f"{path}/reads"
    files = os.listdir(reads_dir)
    for dir in [f"{outdir}/trimmed", f"{outdir}/merged", f"{outdir}/stats", f"{outdir}/filtered", f"{outdir}/derep"]:
        try:
            os.mkdir(dir)
        except FileExistsError:
            print(f"Skipping making {dir}, already exists")

    # Shared arguments for all _exec calls:
    exec_args = {"verbose": verbose, "outdir": outdir, "path": path}

    if log:
        exec_args.update(
            {"logfile": f"{outdir}/log.txt", "errfile": f"{outdir}/err.txt", "cmdfile": f"{outdir}/commands.txt"}
        )

    else:
        exec_args.update({"logfile": None, "errfile": None, "cmdfile": None})

    if readcounts_path is None:
        _exec(f"echo 'left:\t{adapter1}\nright:\t{adapter2}' > {outdir}/adapters.txt", **exec_args)

        # Step 1: categorize files
        endings = [".fq.gz", ".fastq.gz", ".fq", ".fastq"]
        paired = set()
        unpaired = set()

        for filename in files:
            for suffix in endings:
                if filename.endswith(suffix):
                    # Get prefix
                    prefix = filename.rstrip(suffix)
                    for x in endings:
                        prefix = prefix.rstrip(x)

                    # Check if paired
                    if prefix.endswith("_1") or prefix.endswith("_2"):
                        prefix = prefix[:-2]
                        paired.add((prefix, suffix))  # Note double parens
                    else:
                        unpaired.add((prefix, suffix))

        # Step 2: preprocess reads, merge, dereplicate
        for reads_set, paired_bool in zip([paired, unpaired], [True, False]):
            for prefix, suffix in reads_set:
                _process_sample(
                    prefix=prefix,
                    suffix=suffix,
                    adapter1=adapter1,
                    adapter2=adapter2,
                    in_dir=reads_dir,
                    out_dir=outdir,
                    paired=paired_bool,
                    **exec_args,
                )

        _exec(f"cat {outdir}/derep/* > {outdir}/all.fasta")  # Do not log

    else:
        _exec(f"cp {readcounts_path} {outdir}/all.fasta", **exec_args)

    # Step 3: OTU table
    _exec(
        f"{_VSEARCH} --usearch_global",
        f"{outdir}/all.fasta",
        f"--threads {_N_THREADS}",
        f"--id {cutoff}",
        f"--db {db_fasta_path}",
        f"--otutabout {outdir}/otu_table.tsv",
        **exec_args,
    )

    return True
