import os

# Path stuff
CUTADAPT = "/Users/phil/opt/anaconda3/envs/vsearch/bin/cutadapt"
VSEARCH = "/Users/phil/opt/anaconda3/envs/vsearch/bin/vsearch"

# System variables
N_THREADS = 4
FASTQ_MAX_EE = 1.0
FASTQ_MIN_LEN = 225
FASTQ_MAX_NS = 0

def process_sample(
	prefix : str,
	suffix: str,
	adapter1 : str,
	adapter2 : str,
	in_dir : str,
	out_dir : str,
	paired : bool
	) -> bool:
	""" Paired version """

	# Shared values
	cutadapt_log = f"{out_dir}/stats/{prefix}.cutadapt.log" # Cut-adapt log
	out3 = f"{out_dir}/merged/{prefix}{suffix}" # Merged reads
	out4 = f"{out_dir}/stats/{prefix}.stats" # EE stats
	out5 = f"{out_dir}/filtered/{prefix}.filtered.fasta" # Filtered reads
	out6 = f"{out_dir}/derep/{prefix}.derep.fasta"

	if paired:
		path1 = f"{in_dir}/{prefix}_1{suffix}" # Reads mate pair 1
		path2 = f"{in_dir}/{prefix}_2{suffix}" # Reads mate pair 2
		out1 = f"{out_dir}/trimmed/{prefix}_1{suffix}" # Trimmed mate pair 1
		out2 = f"{out_dir}/trimmed/{prefix}_2{suffix}" # Trimmed mate pair 2

		# Cutadapt part
		os.system(f"{CUTADAPT} -A {adapter1} -G {adapter2} -o {out1} -p {out2} -j {N_THREADS} {path1} {path2} > {cutadapt_log}")

		# Merge pairs
		os.system(f"{VSEARCH} --fastq_mergepairs {out1} --reverse {out2} --threads {N_THREADS} --fastqout {out3} --fastq_eeout")

	else:
		path1 = f"{in_dir}/{prefix}{suffix}"
		out1 = f"{out_dir}/trimmed/{prefix}{suffix}"

		# Cutadapt
		os.system(f"{CUTADAPT} -a {adapter1} -g {adapter2} -o {out1} -j {N_THREADS} {path1} > {cutadapt_log}")
		os.system(f"cp {out1} {out3}")

	# Quality stuff
	os.system(f"{VSEARCH} --fastq_eestats {out3} --output {out4}")
	os.system(f"{VSEARCH} --fastq_filter {out3} --fastq_maxee {FASTQ_MAX_EE} --fastq_minlen {FASTQ_MIN_LEN} --fastq_maxns {FASTQ_MAX_NS} --fastaout {out5} --fasta_width 0")
	os.system(f"{VSEARCH} --derep_fulllength {out5} --threads {N_THREADS} --strand plus --sizeout --relabel {prefix}. --output {out6} --fasta_width 0")

	return True


def process_samples(
	path : str,
	adapter1 : str, 
	adapter2 : str,
	) -> bool:
	""" Process all samples """

	# Step 0: relevant preconditions
	files = os.listdir(path)
	out_dir = f"{path}/aPTR_out"
	for f in [out_dir, f"{out_dir}/trimmed", f"{out_dir}/merged", f"{out_dir}/stats", f"{out_dir}/filtered", f"{out_dir}/derep"]:
		try:
			os.mkdir(f)
		except FileExistsError:
			print(f"Skipping making {f}, already exists")

	# Step 1: categorize files
	endings = ['.fq.gz', '.fastq.gz', '.fq', '.fastq']
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
					paired.add((prefix, suffix)) # Note double parens
				else:
					unpaired.add((prefix, suffix))

	# Step 2: preprocess reads, merge, dereplicate
	for prefix, suffix in paired:
		process_sample(prefix, suffix, adapter1, adapter2, path, out_dir, paired=True)
	for prefix, suffix in unpaired:
		process_sample(prefix, suffix, adapter1, adapter2, path, out_dir, paired=False)

	# Step 3: process sequences together
	os.system(f"cat {out_dir}/derep/* > {out_dir}/all.fasta")
	os.system(f"{VSEARCH} --derep_fulllength {out_dir}/all.fasta --threads {N_THREADS} --strand plus --sizein --sizeout --output {out_dir}/all.derep.fasta --fasta_width 0")
	os.system(f"{VSEARCH} --cluster_size {out_dir}/all.derep.fasta --threads {N_THREADS} --id 1.0 --strand plus --sizein --sizeout --centroids {out_dir}/all.centroids.fasta --fasta_width 0")
	os.system(f"{VSEARCH} --sortbysize {out_dir}/all.centroids.fasta --sizein --sizeout --minsize 2 --output {out_dir}/all.sorted.fasta --fasta_width 0")
	os.system(f"{VSEARCH} --uchime_denovo {out_dir}/all.sorted.fasta --sizein --sizeout --fasta_width 0 --qmask none --nonchimeras {out_dir}/all.nonchimeras.fasta")
	os.system(f"{VSEARCH} --usearch_global {out_dir}/all.nonchimeras.fasta --threads {N_THREADS} --id 1.0 --db {out_dir}/../db.fasta --otutabout {out_dir}/all.tsv")
	os.system(f"{VSEARCH} --usearch_global {out_dir}/all.fasta --threads {N_THREADS} --id 1.0 --db {out_dir}/../db.fasta --otutabout {out_dir}/all.tsv")

	return True