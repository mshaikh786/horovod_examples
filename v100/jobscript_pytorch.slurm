#!/bin/bash
#SBATCH --job-name=torch_v100
#SBATCH --ntasks=4
#SBATCH --tasks-per-node=4

## In case Datadataloader needs multiple CPUs for parallel reading from disk
#SBATCH --cpus-per-task=1

#SBATCH --gres=gpu:4
#SBATCH --constraint=v100,gpu_ai
#SBATCH --time=01:00:00

module load cuda/10.2.89
module load dl
module load pytorch/1.5.1 torchvision
module load horovod/0.20.3
module list


export OMPI_MCA_btl_openib_warn_no_device_params_found=0
export UCX_MEMTYPE_CACHE=n
export UCX_TLS=cuda_copy,gdr_copy,sm,tcp,self
if [ ! -d "./data" ]; then
	../download_data.sh
fi
srun -u -n $SLURM_NTASKS -c ${SLURM_CPUS_PER_TASK} --cpu-bind=cores python3 ../pytorch_cifar10.py --batch-size 32 --epoch 10





