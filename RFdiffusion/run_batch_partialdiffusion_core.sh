#!/bin/bash
#SBATCH --job-name=RFdiffusion            # appears in `squeue`
#SBATCH --partition=3090-gcondo      # change to your GPU/CPU partition
#SBATCH --exclude=gpu[2607-2609]
#SBATCH --gres=gpu:1                     # number / type of GPUs per run
#SBATCH --cpus-per-task=1                # adjust to match DataLoader workers
#SBATCH --mem=16G                        # or whatever your data need
#SBATCH --time=24:00:00                  # wall‑clock limit
#SBATCH --output=logs/%x_%A_%a.out       # one log per seed
#SBATCH --array=0-99                     

# ---------- 1.  Environment ----------

# module purges
# module load cuda/12.2                    # or your site’s CUDA module
# source ~/envs/pytorch/bin/activate       # activate the venv/conda env
source ./SE3nv.venv/bin/activate

# ---------- 2.  Pick the seed ----------
SEED=${SLURM_ARRAY_TASK_ID}
PARTIAL_T=${1:-1}

# ---------- 3.  Run your code ----------

echo "→ partial_T      = ${PARTIAL_T}"

./scripts/run_inference.py \
    inference.output_prefix="data/RFdiffusion_output/t${PARTIAL_T}/${SEED}/esm3_abl1b_complete_partialdiffusion_core" \
    inference.input_pdb=data/ESM_original/esm3_abl1b_complete.pdb \
    'contigmap.contigs=["A1-228/287/A516-1149"]' \
    inference.num_designs=1 \
    diffuser.partial_T="${PARTIAL_T}"