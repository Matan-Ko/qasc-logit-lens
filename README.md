# QASC Per-Layer Logit-Lens Study (NLP 0368-3077, 2025/26)

Author: Matan Koshilevitch (ID 211822804) · `koshilevitch@mail.tau.ac.il`

## What this is

Extension of the course's decoder-unblocking starter notebook. We fine-tune
GPT-2 small on QASC and use the logit lens to find the layer at which the
fine-tuned model already "knows" the answer.

**Headline result:** final-layer accuracy on 300 QASC validation examples
goes from **10.67%** (near random 12.5%) to **99.67%** after 3 epochs of
fine-tuning; the answer signal emerges sharply between block 8 and block 9
of 12 (see `outputs/per_layer_accuracy.pdf`).

## Layout

```
qasc_project/
├── code/
│   ├── qasc_decoder_layers_finetuned.py   # main script
│   ├── run_qasc.sbatch                     # full training Slurm job
│   └── run_qasc_sanity.sbatch              # overfit-20 sanity job
├── outputs/
│   ├── per_layer_accuracy.pdf / .png       # the figure
│   ├── acc_before.npy, acc_after.npy       # raw per-layer accuracies
│   └── training_log.out                    # Slurm stdout of the real run
├── report/
│   ├── main.tex                            # ACL-format report
│   ├── refs.bib                            # bibliography
│   └── per_layer_accuracy.pdf              # figure for the report
└── README.md
```

## Reproducing the result on the TAU CS Slurm cluster

One-time setup (from the login node `slurm-client.cs.tau.ac.il`):

```bash
USER_STORE=/vol/joberant_nobck/data/NLP_368307701_2526a/$USER
cd $USER_STORE

# Miniconda in course storage (NOT in $HOME — 4 GB quota).
wget https://repo.anaconda.com/miniconda/Miniconda3-py311_24.7.1-0-Linux-x86_64.sh -O mc.sh
bash mc.sh -b -p $USER_STORE/miniconda3 && rm mc.sh

# Redirect temp/cache AWAY from /tmp (often full on c-00x) before installing torch.
export TMPDIR=$USER_STORE/.tmp PIP_CACHE_DIR=$USER_STORE/.cache/pip
mkdir -p $TMPDIR $PIP_CACHE_DIR

source $USER_STORE/miniconda3/etc/profile.d/conda.sh
conda create -y -n qasc python=3.11
conda activate qasc
pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 torch==2.4.1
pip install --no-cache-dir transformers datasets accelerate matplotlib tqdm
```

Submit the job:

```bash
mkdir -p $USER_STORE/qasc_probe
cp code/qasc_decoder_layers_finetuned.py code/run_qasc.sbatch $USER_STORE/qasc_probe/
cd $USER_STORE/qasc_probe
sbatch run_qasc.sbatch
# wall time ~7 minutes on a Titan XP
```

Outputs land in `$USER_STORE/qasc_probe/outputs/`.

## Reproducing locally (tiny sanity check only — no GPU needed)

The overfit-20 sanity job will fit on any machine with ~2 GB RAM. It does
not produce the paper's main plot but verifies the training loop:

```bash
pip install torch transformers datasets accelerate matplotlib tqdm
python code/qasc_decoder_layers_finetuned.py \
    --train_size 20 --epochs 20 --eval_size 50 --out_dir sanity_out
# training loss should drop from ~6.7 to <0.01
```

## Building the report PDF

`pdflatex` is not assumed to be installed locally. Fastest option:

1. Go to <https://overleaf.com>, start a **blank** project, delete its
   `main.tex`.
2. Upload `report/main.tex`, `report/refs.bib`, and
   `report/per_layer_accuracy.pdf`.
3. Also upload the ACL style file `acl.sty` and the bibliography style
   `acl_natbib.bst` from <https://github.com/acl-org/acl-style-files> — or,
   more simply, start the Overleaf project from the template gallery
   "ACL 2023" (includes both files) and then replace only `main.tex`,
   `refs.bib`, and the figure.
4. Compile with pdfLaTeX. Run BibTeX if the references do not appear
   (Overleaf's default build chain handles this automatically).
