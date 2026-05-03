#!/bin/bash
#SBATCH --job-name=qasc-probe
#SBATCH --partition=studentkillable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=180
#SBATCH --signal=USR1@120
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# Point HuggingFace + pip caches at the shared per-user storage so the
# home-directory quota does not fill up (see course cluster guide).
USER_STORE="/vol/joberant_nobck/data/NLP_368307701_2526a/${USER}"
export HF_HOME="${USER_STORE}/.cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export PIP_CACHE_DIR="${USER_STORE}/.cache/pip"
mkdir -p "${HF_HOME}" "${PIP_CACHE_DIR}"

# Activate the qasc conda env
source "${USER_STORE}/miniconda3/etc/profile.d/conda.sh"
conda activate qasc

cd "${USER_STORE}/qasc_probe"

python qasc_decoder_layers_finetuned.py \
    --model_name gpt2 \
    --eval_size 300 \
    --epochs 3 \
    --batch_size 8 \
    --lr 5e-5 \
    --out_dir outputs
