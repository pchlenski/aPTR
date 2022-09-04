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


def _exec(*args, verbose=False, cmdfile=None, logfile=None, errfile=None, **kwargs):
    """Utility function to log and execute system calls the way I like it"""
    # if isinstance(cmd, list):
    #     cmd = [str(x) for x in cmd]
    #     cmd = " ".join(cmd)

    # Coerce to string
    cmd = " ".join([str(x) for x in args])

    if verbose and cmdfile is not None:
        with open(cmdfile, "a") as f:
            print(cmd, file=f)
    elif verbose:
        print(cmd)

    if logfile is not None:
        cmd = f"{cmd} >> {logfile}"

    if errfile is not None:
        cmd = f"{cmd} 2>> {errfile}"

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
    **exec_args,
) -> bool:
    """For a single FASTQ/pair of FASTQ files, cut adapters, filter, and dereplicate"""

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
                **exec_args,
            )
        else:
            _exec(f"cp {path1} {out1}", **exec_args)
            _exec(f"cp {path2} {out2}", **exec_args)

        # Merge pairs
        _exec(
            f"{_VSEARCH} --fastq_mergepairs {out1} --reverse {out2} --threads {_N_THREADS} --fastqout {out3} --fastq_eeout",
            **exec_args,
        )

    # For unpaired files, simply cut adapters
    else:
        path1 = f"{in_dir}/{prefix}{suffix}"
        out1 = f"{out_dir}/trimmed/{prefix}{suffix}"

        # Cutadapt
        if use_cutadapt:
            _exec(
                f"{_CUTADAPT} -a {adapter1} -g {adapter2} -o {out1} -j {_N_THREADS} {path1} > {cutadapt_log}",
                **exec_args,
            )
        else:
            _exec(f"cp {path1} {out1}", **exec_args)

        # Just copy rather than merging
        _exec(f"cp {out1} {out3}", **exec_args)

    # Quality stuff
    try:
        _exec(
            _VSEARCH,
            "--fastq_eestats2",
            out3,
            "--output",
            out4,
            "--fastq_qmax",
            _FASTQ_QMAX,
            **exec_args,
        )
        _exec(
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
            0,
            **exec_args,
        )
        _exec(
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
            0,
            **exec_args,
        )
    except Exception:
        pass

    return True


def process_samples(
    path: str,
    adapter1: str,
    adapter2: str,
    db_path: str,
    outdir: str,
    verbose: bool = True,
    log: bool = True,
) -> bool:
    """Process all samples in a directory"""

    # Step 0: relevant preconditions
    files = os.listdir(path)
    for dir in [
        f"{outdir}/trimmed",
        f"{outdir}/merged",
        f"{outdir}/stats",
        f"{outdir}/filtered",
        f"{outdir}/derep",
    ]:
        try:
            os.mkdir(dir)
        except FileExistsError:
            print(f"Skipping making {dir}, already exists")

    # Shared arguments for all _exec calls:
    exec_args = {"verbose": verbose, "outdir": outdir, "path": path}

    if log:
        exec_args.update(
            {
                "logfile": f"{outdir}/log.txt",
                "errfile": f"{outdir}/err.txt",
                "cmdfile": f"{outdir}/commands.txt",
            }
        )

    else:
        exec_args.update({"logfile": None, "errfile": None, "cmdfile": None})

    _exec(
        f"echo 'left:\t{adapter1}\nright:\t{adapter2}' > {outdir}/adapters.txt"
    )  # Do not log this

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
    for reads_set, paired_bool in zip([paired, unpaired], [True, False]):
        for prefix, suffix in reads_set:
            _process_sample(
                prefix=prefix,
                suffix=suffix,
                adapter1=adapter1,
                adapter2=adapter2,
                in_dir=path,
                out_dir=outdir,
                paired=paired_bool,
                **exec_args,
            )

    # Step 3: OTU table
    _exec(f"cat {outdir}/derep/* > {outdir}/all.fasta")  # Do not log
    _exec(
        f"{_VSEARCH} --usearch_global",
        f"{outdir}/all.fasta",
        f"--threads {_N_THREADS}",
        f"--id 1.0",
        f"--db {db_path}",
        f"--otutabout {outdir}/all.tsv",
        **exec_args,
    )

    return True
