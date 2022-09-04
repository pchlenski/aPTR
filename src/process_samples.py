""" Scripts for running VSEARCH and CUTADAPT on FASTRQ reads """

import os

# Path stuff
_CUTADAPT = "cutadapt"
_VSEARCH = "vsearch"

# System variables
_N_THREADS = 4
_FASTQ_MAX_EE = 1.0
_FASTQ_MIN_LEN = 225
_FASTQ_MAX_NS = 0
_FASTQ_QMAX = 93


def _exec(cmd, verbose, logfile=None, errfile=None, cmdfile=None):
    """ Utility function to log and execute system calls the way I like it """
    if isinstance(cmd, list):
        cmd = [str(x) for x in cmd]
        cmd = " ".join(cmd)
    if verbose:
        if cmdfile is not None:
            with open(cmdfile, "w") as f:
                print(cmd, file=f)
        else:
            print(cmd)
    if logfile is not None:
        cmd = f"{cmd} > {logfile}"
    if errfile is not None:
        cmd = f"{cmd} 2> {errfile}"
    out = os.system(cmd)
    if out != 0:
        print(f"Failed to execute command:\t{cmd}")
        raise Exception(f"out status is {out}")


def _process_sample(
    prefix: str,
    suffix: str,
    adapter1: str,
    adapter2: str,
    in_dir: str,
    out_dir: str,
    paired: bool,
    verbose: bool,
    logfile: str,
    errfile: str,
    cmdfile: str,
) -> bool:
    """ For a single FASTQ/pair of FASTQ files, cut adapters, filter, and dereplicate """

    # Shared values
    cutadapt_log = f"{out_dir}/stats/{prefix}.cutadapt.log"  # Cut-adapt log
    out3 = f"{out_dir}/merged/{prefix}{suffix}"  # Merged reads
    out4 = f"{out_dir}/stats/{prefix}.stats"  # EE stats
    out5 = f"{out_dir}/filtered/{prefix}.filtered.fasta"  # Filtered reads
    out6 = f"{out_dir}/derep/{prefix}.derep.fasta"

    use_cutadapt = (adapter1 is not None and adapter2 is not None) and (
        adapter1 != "" and adapter2 != ""
    )

    # For paired files: cut adapters individually, then merge pairs
    if paired:
        path1 = f"{in_dir}/{prefix}_1{suffix}"  # Reads mate pair 1
        path2 = f"{in_dir}/{prefix}_2{suffix}"  # Reads mate pair 2
        out1 = f"{out_dir}/trimmed/{prefix}_1{suffix}"  # Trimmed mate pair 1
        out2 = f"{out_dir}/trimmed/{prefix}_2{suffix}"  # Trimmed mate pair 2

        # Cutadapt part
        if use_cutadapt:
            _exec(
                f"{_CUTADAPT} -A {adapter1} -G {adapter2} -o {out1} -p {out2} -j {_N_THREADS} {path1} {path2} > {cutadapt_log}",
                verbose,
                logfile,
                errfile,
                cmdfile,
            )
        else:
            _exec(f"cp {path1} {out1}", verbose, logfile, errfile, cmdfile)
            _exec(f"cp {path2} {out2}", verbose, logfile, errfile, cmdfile)

        # Merge pairs
        _exec(
            f"{_VSEARCH} --fastq_mergepairs {out1} --reverse {out2} --threads {_N_THREADS} --fastqout {out3} --fastq_eeout",
            verbose,
            logfile,
            errfile,
            cmdfile,
        )

    # For unpaired files, simply cut adapters
    else:
        path1 = f"{in_dir}/{prefix}{suffix}"
        out1 = f"{out_dir}/trimmed/{prefix}{suffix}"

        # Cutadapt
        if use_cutadapt:
            _exec(
                f"{_CUTADAPT} -a {adapter1} -g {adapter2} -o {out1} -j {_N_THREADS} {path1} > {cutadapt_log}",
                verbose,
                logfile,
                errfile,
                cmdfile,
            )
        else:
            _exec(f"cp {path1} {out1}", verbose, logfile, errfile, cmdfile)

        # Just copy rather than merging
        _exec(f"cp {out1} {out3}", verbose, logfile, errfile, cmdfile)

    # Quality stuff
    try:
        _exec(
            [
                _VSEARCH,
                "--fastq_eestats2",
                out3,
                "--output",
                out4,
                "--fastq_qmax",
                _FASTQ_QMAX,
            ],
            verbose,
            logfile,
            errfile,
            cmdfile,
        )
        _exec(
            [
                _VSEARCH,
                "--fastq_filter",
                out3,
                "--fastq_maxee",
                _FASTQ_MAX_EE,
                "--fastq_minlen",
                _FASTQ_MIN_LEN,
                "--fastq_maxns",
                _FASTQ_MAX_NS,
                "--fastaout",
                out5,
                "--fastq_qmax",
                _FASTQ_QMAX,
                "--fasta_width",
                "0",
            ],
            verbose,
            logfile,
            errfile,
            cmdfile,
        )
        _exec(
            [
                _VSEARCH,
                "--derep_fulllength",
                out5,
                "--strand",
                "plus",
                "--sizeout",
                "--relabel",
                f"{prefix}.",
                "--output",
                out6,
                "--fasta_width",
                "0",
            ],
            verbose,
            logfile,
            errfile,
            cmdfile,
        )
    except Exception:
        pass

    return True


def _process_samples(
    path: str,
    adapter1: str,
    adapter2: str,
    db_path: str,
    verbose: bool = True,
    log: bool = True,
) -> bool:
    """ Process all samples in a directory """

    # Step 0: relevant preconditions
    files = os.listdir(path)
    out_dir = f"{path}/aPTR_out"
    for f in [
        out_dir,
        f"{out_dir}/trimmed",
        f"{out_dir}/merged",
        f"{out_dir}/stats",
        f"{out_dir}/filtered",
        f"{out_dir}/derep",
    ]:
        try:
            os.mkdir(f)
        except FileExistsError:
            print(f"Skipping making {f}, already exists")

    if log:
        logfile = f"{out_dir}/log.txt"
        errfile = f"{out_dir}/err.txt"
        cmdfile = f"{out_dir}/commands.txt"

    else:
        logfile = None
        errfile = None
        cmdfile = None

    _exec(
        f"echo 'left:\t{adapter1}\nright:\t{adapter2}' > {out_dir}/adapters.txt",
        verbose,
        logfile,
        errfile,
        cmdfile,
    )

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
                    prefix = prefix.rstrip("_1").rstrip("_2")
                    paired.add((prefix, suffix))  # Note double parens
                else:
                    unpaired.add((prefix, suffix))

    # Step 2: preprocess reads, merge, dereplicate
    for prefix, suffix in paired:
        _process_sample(
            prefix,
            suffix,
            adapter1,
            adapter2,
            path,
            out_dir,
            paired=True,
            verbose=verbose,
            logfile=logfile,
            errfile=errfile,
            cmdfile=cmdfile,
        )

    for prefix, suffix in unpaired:
        _process_sample(
            prefix,
            suffix,
            adapter1,
            adapter2,
            path,
            out_dir,
            paired=False,
            verbose=verbose,
            logfile=logfile,
            errfile=errfile,
            cmdfile=cmdfile,
        )

    # Step 3: OTU table
    _exec(
        f"cat {out_dir}/derep/* > {out_dir}/all.fasta",
        verbose=verbose,
        logfile=logfile,
        errfile=errfile,
        cmdfile=cmdfile,
    )
    _exec(
        [
            _VSEARCH,
            "--usearch_global",
            f"{out_dir}/all.fasta",
            "--threads",
            _N_THREADS,
            "--id 1.0",
            "--db",
            db_path,
            "--otutabout",
            f"{out_dir}/all.tsv",
        ],
        verbose=verbose,
        logfile=logfile,
        errfile=errfile,
        cmdfile=cmdfile,
    )

    return True
