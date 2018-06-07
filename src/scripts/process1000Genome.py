import os
import pandas as pd
import argparse 
import numpy as np
'''
python helper to generate/read the data
'''
ap = argparse.ArgumentParser()

ap.add_argument('--vcfFileName',type=str)
ap.add_argument('--ancestryDir',type=str)
ap.add_argument('--chromosomeNum',type=str)
ap.add_argument('--processedDir',type=str)

args = ap.parse_args()

vcfFileName = args.vcfFileName
ancestryDir = args.ancestryDir
chromosomeNum = args.chromosomeNum
processedDir = args.processedDir

#TODO check if file exists
#skip vcf header
df = pd.read_csv(vcfFileName,skiprows=29,sep='\t')
#rename first column to remove #delimiter
df.rename(columns={df.columns[0]:df.columns[0][1:]},inplace=True)

#drops all columns except for chromosome, position, and individuals
df.drop(df.columns[2:9],axis=1,inplace=True)

#computing the genotype from string
string2Genome = lambda string : int(string[0]) + int(string[2])
df[df.columns[2:]] = df[df.columns[2:]].applymap(string2Genome) 

#for every individual
dfLabels = df.copy(deep=True)
class AncestryMap:
    def __init__(self,indiv):
        ancestorRange = pd.read_csv(\
                os.path.join(ancestryDir, indiv+".bed"),delimiter='\t')
        def pop2Int(pop):
            if pop == 'undet':
                return -1
            else:
                return int(pop)
        ancestorRange[ancestorRange.columns[3]] = ancestorRange[ancestorRange.columns[3]]\
                .apply(pop2Int)
        ancestorRange = ancestorRange.values 
        self.chromosomes = ancestorRange[:,0]
        self.endPos = ancestorRange[:,2]
        self.population = ancestorRange[:,3]
    def getPopulation(self,chrom,pos):
        startIdx = np.searchsorted(self.chromosomes,chrom)
        endIdx = np.searchsorted(self.chromosomes,chrom,side='right')
        offset = np.searchsorted(self.endPos[startIdx:endIdx],pos)
        return self.population[startIdx+offset]
for indiv in dfLabels.columns[2:]:
    #read in an
    currAncestryMap = AncestryMap(indiv)
    getPopWrapperFn = lambda args : currAncestryMap.getPopulation(args[0],args[1])
    dfLabels[indiv] = dfLabels[dfLabels.columns[0:2]].apply(getPopWrapperFn,axis=1)
#dropping chrom and pos
#dfLabels.drop(dfLabels.columns[0:2],axis=1,inplace=True)

#saving file
df.to_csv(os.path.join(processedDir,"genotypes.csv"))
dfLabels.to_csv(os.path.join(processedDir,"ancestors.csv"))

