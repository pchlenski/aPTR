# aPTR: amplicon peak-to-trough ratio
Microbial dynamics inferred from single-sample amplicon sequencing

## Generating tables
We used [PATRIC p3-scripts](https://github.com/PATRIC3/PATRIC-distribution) to grab all 16S and dnaA from the PATRIC database.

Command to generate `allSSU.tsv`:

```bash
p3-all-genomes \
    --eq reference_genome,Representative \
    --attr genome_id,contigs,genome_length \
| p3-get-genome-features \
    --col 1 \
    --eq feature_type,rRNA \ 
    --eq product,SSU \
    --attr accession,start,end,strand,patric_id,na_sequence,na_sequence_md5,product \
> allSSU.tsv
```

Command to generate `allDnaA.tsv`:

```bash
p3-all-genomes \
    --eq reference_genome,Representative \
    --attr genome_id,contigs,genome_length \
| p3-get-genome-features \
    --col 1 \
    --eq product,dnaA \
    --attr accession,start,end,strand,patric_id,product \
> allDnaA.tsv
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
