#!/bin/sh

#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=10:00:00
#SBATCH --partition=gtx1080
#SBATCH --gres=gpu:2

module load apptainer

apptainer exec --nv tensorflow_latest.sif python attentive_fp.py
