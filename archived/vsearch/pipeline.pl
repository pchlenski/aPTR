use strict;
use warnings;
use Data::Dumper;

my $path = "./out";
my $reads = "/home/phil/DATA/16s/PRJNA695570/16s_downsample";

open my $metadata_fh, "<", "$path/metadata.txt";

<$metadata_fh>; # this skips header

# preprocessing steps:
`mkdir $path/trimmed`;
`mkdir $path/merged`;
`mkdir $path/stats`;
`mkdir $path/filtered`;
`mkdir $path/derep`;

# individual file operations
while (<$metadata_fh>){
	chomp $_;
	my ($row_id, $sample, $wgs, $amp) = split /\s+/, $_;

	# trim -- use conda env 'cutadapt'
	`cutadapt -g GTGYCAGCMGCCGCGGTAA -G CCGYCAATTYMTTTRAGTTT -o $path/trimmed/$amp\_pass_1.fastq -p $path/trimmed/$amp\_pass_2.fastq $reads/$amp\_pass_1.fastq.gz $reads/$amp\_pass_2.fastq.gz`;

	# merge -- use conda env 'vsearch' for everything else
	`vsearch --fastq_mergepairs $path/trimmed/$amp\_pass_1.fastq --reverse $path/trimmed/$amp\_pass_2.fastq --threads 12 --fastqout $path/merged/$amp.merged.fastq --fastq_eeout`;

	# quality stuff
	`vsearch --fastq_eestats $path/merged/$amp.merged.fastq --output $path/stats/$amp.stats`;
	`vsearch --fastq_filter $path/merged/$amp.merged.fastq --fastq_maxee 1.0 --fastq_minlen 225 --fastq_maxns 0 --fastaout $path/filtered/$amp.filtered.fasta --fasta_width 0`;
	`vsearch --derep_fulllength $path/filtered/$amp.filtered.fasta --threads 12 --strand plus --sizeout --relabel $amp. --output $path/derep/$amp.derep.fasta --fasta_width 0`;
}

close $metadata_fh;

# merged operations
`cat $path/derep/* > $path/all.fasta`;
`vsearch --usearch_global $path/all.fasta --threads 12 --id 1.0 --db $path/../db.fasta --otutabout $path/all.tsv`;
