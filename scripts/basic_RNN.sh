#!/bin/sh
#$ -cwd
#$ -j y
#$ -e ./stdout.log 
#$ -o ./stderr.log
#$ -l h_data=4G, h_rt =23:00:00
#$ -t 1-200:1

SGE_TASK_ID=1

source /u/local/Modules/default/init/modules.sh
module load python/3.6.1

for i in {1..200}
do
    COUNTER=$((COUNTER+1))
    if [[ $COUNTER -eq $SGE_TASK_ID ]]
    then
        python src/main.py
    fi
done

