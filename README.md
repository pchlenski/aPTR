# aPTR: amplicon peak-to-trough ratio
Microbial dynamics inferred from single-sample amplicon sequencing

## Generating tables
We used [PATRIC p3-scripts](https://github.com/PATRIC3/PATRIC-distribution) to grab all 16S and dnaA from the PATRIC database.

Command to generate `patric_table.tsv`:

```bash
p3-all-genomes \
    --eq reference_genome,Representative \
    --attr genome_id,contigs,genome_length,genome_name \
| p3-get-genome-features --col 1 \
    --eq "product,Chromosomal replication initiator protein DnaA" \
    --attr accession,start,end,strand,patric_id,product \
| p3-get-genome-features \
    --col 1 \
    --eq feature_type,rRNA \
    --eq product,SSU \
    --attr accession,start,end,strand,patric_id,na_sequence,na_sequence_md5,product \
> patric_table.tsv
```

## Downloading sequences
All sequences were downloaded from PATRIC FTP site. TODO: Add notebook to do this.

## PTR inference with coPTR
To compare this tool to coPTR, we ran the script `simulate_complete_genomes.py` with the following settings:
* `n_samples`: 100
* `n_genomes_per_sample`: 20
* `scale`: 1e5

The output directory was moved to the directory `./out/complete_1e5`. Then to obtain coPTR-based PTR estimates, we used the following commands:
```bash
# install coPTR
git clone https://www.github.com/tyjo/coptr
cd coptr
conda env create -f coptr.yml
conda activate coptr
pip install .

# directory management
mkdir ~/coptr_16s
mkdir ~/coptr_16s/dbs
mkdir ~/coptr_16s/dbs/draft
mkdir ~/coptr_16s/dbs/complete
mkdir ~/coptr_16s/bam
mkdir ~/coptr_16s/coverage
cd ~/16s-ptr

# generate coPTR index for all genomes
cp -r ./data/seqs ./data/seqs_unzipped
gunzip ./data/seqs_unzipped
coptr index ./data/seqs_unzipped ~/coptr_16s/dbs/draft/index
python ./filter_to_complete_genomes.py
gunzip ./data/seqs_complete
coptr index ./data/seqs_complete ~/coptr_16s/dbs/complete/index

# read mapping
coptr map ~/coptrs_16s/dbs/draft/index ./out/complete_1e5 ~/coptr_16s/bam
```

## VSEARCH pipeline for generating OTUs
It is important to process FASTQ reads directly because aPTR does not tolerate clustering OTUs. Only exact subsequence matches are allowed. For this study, we used PRJNA695570: Human gut metagenomes of healthy mothers and their children. According to NCBI SRA, the samples are prepared as follows:

>Design: Amplicon fragments are PCR-amplified from the DNA in duplicate using separate template dilutions using the high-fidelity Phusion polymerase. A single round of PCR was done using "fusion primers" (Illumina adaptors + indices + specific regions) targeting the V4V5 region of the 16S rRNA gene (515FB=GTGYCAGCMGCCGCGGTAA and 926R=CCGYCAATTYMTTTRAGTTT). PCR products were verified visually by running on a high-throughput Hamilton Nimbus Select robot using Coastal Genomics Analytical Gels. The PCR reactions from the same samples are pooled in one plate, then cleaned-up and normalized using the high-throughput Charm Biotech Just-a-Plate 96-well Normalization Kit. Up to 380 samples were then pooled to make one library which was then quantified fluorometrically before sequencing.

Accordingly, the pipeline (which is automated via Perl script in `vsearch/pipeline.pl`) was constructed as follows:

For each sample:
```bash
# Remove adapter contamination
cutadapt \
    -g GTGYCAGCMGCCGCGGTAA \
    -G CCGYCAATTYMTTTRAGTTT \
    -o ./trimmed/$SAMPLE_pass_1.fastq \
    -p ./trimmed/$SAMPLE_pass_2.fastq \
    ./reads/$SAMPLE_pass_1.fastq.gz \
    ./reads/$SAMPLE_pass_2.fastq.gz

# Merge paired-end FASTQ reads
vsearch --fastq_mergepairs ./trimmed/$SAMPLE_pass_1.fastq \
    --reverse ./trimmed/$SAMPLE_pass_2.fastq \
    --threads 12 \
    --fastqout ./merged/$SAMPLE.merged.fastq \
    --fastq_eeout

# Quality metrics, filtering, and dereplication
vsearch --fastq_eestats ./merged/$SAMPLE.merged.fastq --output ./stats/$SAMPLE.stats
vsearch --fastq_filter ./merged/$SAMPLE.merged.fastq \
    --fastq_maxee 1.0 \
    --fastq_minlen 225 \
    --fastq_maxns 0 \
    --fastaout ./filtered/$SAMPLE.filtered.fasta \
    --fasta_width 0
vsearch --derep_fulllength ./filtered/$SAMPLE.filtered.fasta \
    --threads 12 \
    --strand plus \
    --sizeout \
    --relabel $SAMPLE. \
    --output ./derep/$SAMPLE.derep.fasta \
    --fasta_width 0
```

Then, the samples are merged and processed into a single OTU table:
```bash
cat ./derep/* > ./all.fasta
vsearch --usearch_global path/all.fasta \
    --threads 12 \
    --id 1.0 \
    --db ../db.fasta \
    --otutabout ./all.tsv
```

## Guide to various scripts
`aptr.py`: My draft of a script for running aPTR. You need to point it at a 
directory containing a `reads` subdirectory and give it your 16S primers, 
and it will generate an OTU table and database for you. If you already have some
of those things, use the flags `--db_path` and `otu_path` to specify a database (fasta file).

`get_sequences_from_patric.py` downloads all genomes in the large OTU database. This will take up 1.3 GB of space.

`simulate_complete_genomes.py` simulates genomes from those sequences in analogous FASTQ files corresponding to WGS reads and an OTU matrix showing just the in-frame 16S hits. 