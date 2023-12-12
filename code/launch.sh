#!/bin/bash

##SBATCH --job-name=dmr
#SBATCH --output=slurm-dmr_%j.out
##SBATCH --error=slurm-dmr_%j.err

export PATH=$SLURM_ROOT/bin:$PATH
NODELIST="$(scontrol show hostname $SLURM_JOB_NODELIST | paste -d, -s)"
#echo $NODELIST

mpirun -n $SLURM_JOB_NUM_NODES -hosts $NODELIST ./jacobi

