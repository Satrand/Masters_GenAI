#!/bin/bash 

#SBATCH --export=ALL
#SBATCH --partition=interruptible_gpu
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --time=72:00:00
#SBATCH --mem=20G 
#SBATCH --job-name=cvae_training
#SBATCH --no-requeue
#SBATCH --mail-user=sara.strand@kcl.ac.uk
#SBATCH --mail-type=START,END,FAIL

#SBATCH --output=/scratch/prj/rcmb_genai_transition/CVAE_Stest/%j.out   # Logs to scratch 
#SBATCH --error=/scratch/prj/rcmb_genai_transition/CVAE_Stest/%j.err    # Separate stderr
#======================================================

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

module purge

module load cudnn/8.7.0.84-11.8-gcc-13.2.0 
module load cuda/12.2.1-gcc-13.2.0

#======================================================

# Activate Python virtual enviroment - change to your path.  
source /scratch/prj/rcmb_genai_transition/venv_torch/bin/activate

python3 /scratch/prj/rcmb_genai_transition/CVAE_Stest/train.py --config 50k_config.yaml > output 
