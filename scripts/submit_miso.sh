#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=miso
#SBATCH -t 600:00:00
#SBATCH -p preemptable
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --exclude=esplhpc-cp018
#SBATCH --mem-per-gpu=256G
#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --mail-user=tyler.lam@cshs.org
#SBATCH --chdir=/common/lamt2/miso_rapids/miso

LD_LIBRARY_PATH=/common/lamt2/miniforge3/envs/miso_rapids/lib:$LD_LIBRARY_PATH
source /common/lamt2/miniforge3/bin/activate miso_rapids

log=$1
args=${@:2}

/common/lamt2/miniforge3/envs/miso_rapids/bin/python -u scripts/run_miso.py ${args} >& out_miso_${log}.out