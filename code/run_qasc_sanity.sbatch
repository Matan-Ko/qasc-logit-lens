#!/bin/bash
#SBATCH --job-name=qasc-sanity
#SBATCH --partition=studentkillable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --time=30
#SBATCH --signal=USR1@120
#SBATCH --output=sanity-%j.out
#SBATCH --error=sanity-%j.err

USER_STORE="/vol/joberant_nobck/data/NLP_368307701_2526a/${USER}"
export HF_HOME="${USER_STORE}/.cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export PIP_CACHE_DIR="${USER_STORE}/.cache/pip"
mkdir -p "${HF_HOME}" "${PIP_CACHE_DIR}"

source "${USER_STORE}/miniconda3/etc/profile.d/conda.sh"
conda activate qasc

cd "${USER_STORE}/qasc_probe"

# Sanity: can the loop overfit 20 examples? If not, the training code is broken.
# Expected: training loss drops near 0, post-FT final-layer accuracy on the 50-eval
# subset jumps well above the pre-FT ~12.5% baseline.
python qasc_decoder_layers_finetuned.py \
    --model_name gpt2 \
    --train_size 20 \
    --epochs 20 \
    --eval_size 50 \
    --batch_size 4 \
    --lr 5e-5 \
    --out_dir outputs_sanity
