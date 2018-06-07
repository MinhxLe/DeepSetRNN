#!/bin/sh

####settings####
POP_NAME=$1
CHROM_NUM=$2
BP_NUM=$3

BASE_DIR='/u/home/m/minhle/project-sriram/minh/data'
VCF_DIR=$BASE_DIR'/1000genome/phase1/integrated_call_sets'
ANCESTRY_DIR=$BASE_DIR'/1000genome/phase1/ancestry_deconvolution'
POP_DIR=$ANCESTRY_DIR'/'$POP_NAME
PROCESSED_DIR=$BASE_DIR'/DeepSetProcessed/'$POP_NAME"_CHR"$CHROM_NUM"_BP"$BP_NUM

mkdir -p $PROCESSED_DIR

#getting labels of individuals
POP_LIST_FNAME=$ANCESTRY_DIR'/'$POP_NAME'_individuals.txt'
if [ ! -f $POP_LIST_FNAME ]; then 
    echo 'generating list of individuals in this population'
    ls $POP_DIR/*.bed | sed  -e 's/.*\/\(.*\)\.bed$/\1/' > $POP_LIST_FNAME
fi

#filtering with vftool the individuals, chromosome and base
VCF_FNAME=$VCF_DIR'/ALL.chr'$CHROM_NUM$'.integrated_phase1_v3.20101123.snps_indels_svs.genotypes.vcf'
VCF_PROCESSED=$PROCESSED_DIR'/filtered'
PROCESSED_FNAME=$VCF_PROCESSED'.recode.vcf'

if [ ! -f $PROCESSED_FNAME ]; then
    echo 'filtering with vcftools'
    vcftools --vcf $VCF_FNAME \
        --recode \
        --out $VCF_PROCESSED \
        --chr $CHROM_NUM \
        --to-bp $BP_NUM \
        --remove-indels \
        --remove-filtered-all \
        --min-alleles 2 --max-alleles 2 \
        --max-missing 1 \
        --keep $POP_LIST_FNAME \
        --remove-filtered-geno-all
fi

##generating the CSV file from vcf file with python scripy
echo 'running python script to generate csv genomes and population labels'
python2 process1000Genome.py --vcfFileName $PROCESSED_FNAME\
    --ancestryDir $POP_DIR --chromosomeNum $CHROM_NUM\
    --processedDir $PROCESSED_DIR
##remove the header sections


