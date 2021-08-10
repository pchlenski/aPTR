# 16s-ptr
Microbial dynamics inferred from single-sample amplicon sequencing

Command to generate `allSSU.tsv`:
```bash
p3-all-genomes --eq reference_genome,Representative --attr genome_id,contigs,genome_length | p3-get-genome-features --col 1 --eq feature_type,rRNA --eq product,SSU --attr accession,start,end,strand,patric_id,na_sequence,na_sequence_md5,product > allSSU.tsv
```

Command to generate `allDnaA.tsv`:
```bash
p3-all-genomes --eq reference_genome,Representative --attr genome_id,contigs,genome_length | p3-get-genome-features --col 1 --eq product,dnaA --attr accession,start,end,strand,patric_id,na_sequence,na_sequence_md5,product > allDnaA.tsv
```