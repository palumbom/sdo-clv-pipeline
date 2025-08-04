#!/bin/bash

# # queue the analysis jobs
# jid1=$(sbatch --parsable /storage/home/mlp95/work/sdo-clv-pipeline/batch/run2012.sh)
# jid2=$(sbatch --parsable /storage/home/mlp95/work/sdo-clv-pipeline/batch/run2013.sh)
# jid3=$(sbatch --parsable /storage/home/mlp95/work/sdo-clv-pipeline/batch/run2014.sh)
# jid4=$(sbatch --parsable /storage/home/mlp95/work/sdo-clv-pipeline/batch/run2015.sh)

# # merge the output
# jid5=$(sbatch --parsable --dependency=afterok:${jid1}:${jid2}:${jid3}:${jid4} /storage/home/mlp95/work/sdo-clv-pipeline/batch/merge_output.sh)
# jid6=$(sbatch --parsable --dependency=afterok:${jid5} /storage/home/mlp95/work/sdo-clv-pipeline/batch/preprocess_output.sh)


# queue the analysis jobs
jid1=$(sbatch --parsable /mnt/home/mpalumbo/work/savannah/sdo-clv-pipeline/batch/runall.sh)

# merge the output
jid2=$(sbatch --parsable --dependency=afterok:${jid1} /mnt/home/mpalumbo/work/savannah/sdo-clv-pipeline/batch/merge_output.sh)
jid3=$(sbatch --parsable --dependency=afterok:${jid2} /mnt/home/mpalumbo/work/savannah/sdo-clv-pipeline/batch/preprocess_output.sh)
