#!/bin/bash

#SBATCH --ntasks=4
#SBATCH --tasks-per-node=4

## In case Datadataloader needs multiple CPUs for parallel reading from disk
#SBATCH --cpus-per-task=1

#SBATCH --gres=gpu:2
#SBATCH --constraint=v100,gpu_ai
#SBATCH --time=00:15:00

module load cuda/10.2.89
module load dl
module load tensorflow/2.2
module load horovod/0.20.3
module list


export OMPI_MCA_btl_openib_warn_no_device_params_found=0
export UCX_MEMTYPE_CACHE=n
export UCX_TLS=cuda_copy,gdr_copy,sm,tcp,self
export NCCL_DEBUG=INFO

srun -u -n $SLURM_NTASKS -c ${SLURM_CPUS_PER_TASK} --cpu-bind=cores python3 ../tensorflow2_keras_mnist.py





