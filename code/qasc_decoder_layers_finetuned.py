"""QASC decoder-layer probing + fine-tuning.

Starts from the course starter notebook (QASC_decoder_layers.ipynb) and adds a
causal-LM fine-tuning step on QASC-train so the layerwise logit-lens probe
actually shows meaningful learning (instead of chance-level 1/8 accuracy).

Pipeline:
  1. Load QASC + GPT-2.
  2. Evaluate per-layer accuracy on a validation subset (pre-FT).
  3. Fine-tune on QASC-train, computing loss only on the final answer-letter
     token (all prompt tokens are masked with -100).
  4. Evaluate per-layer accuracy again (post-FT).
  5. Plot both curves on the same figure + save PDF/PNG.

Runs end-to-end on a single GPU (GPT-2 small is ~124M params).
"""

import argparse
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="gpt2")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval_size", type=int, default=300,
                   help="How many validation examples to probe (per-layer).")
    p.add_argument("--max_len", type=int, default=384,
                   help="Max token length for prompt (+1 for answer letter).")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--out_dir", default="outputs")
    p.add_argument("--skip_train", action="store_true",
                   help="Skip fine-tuning, only produce the pre-FT plot.")
    p.add_argument("--train_size", type=int, default=-1,
                   help="If > 0, subsample the training set. Useful to sanity-check "
                        "that the loop can overfit a tiny slice (loss -> ~0).")
    return p.parse_args()


LETTERS = list("ABCDEFGH")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------
def make_prompt(ex) -> str:
    opts = ex["choices"]["text"]
    opt_str = " ".join(f"({LETTERS[i]}) {opts[i]}" for i in range(8))
    return (
        f"Fact: {ex['combinedfact']}\n"
        f"Q: {ex['question']}\n"
        f"Choices: {opt_str}\n"
        f"Answer:"
    )


def verify_letter_tokens(tok):
    """Each " A".." H" must encode to exactly one token with this tokenizer."""
    ids = {}
    for L in LETTERS:
        enc = tok.encode(" " + L, add_special_tokens=False)
        if len(enc) != 1:
            raise ValueError(f"Letter '{L}' is not 1 token, got {enc}")
        ids[L] = enc[0]
    return ids


# ---------------------------------------------------------------------------
# Layerwise logit-lens evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def layerwise_letter_preds(model, tok, prompt: str, letter_token_ids, device, max_len):
    """For each transformer layer (plus embeddings), apply ln_f + lm_head to the
    hidden state at the last prompt position and pick the argmax letter token.
    Returns a list of predictions, length = num_layers + 1.
    """
    enc = tok(prompt, return_tensors="pt", truncation=True, max_length=max_len)
    enc = {k: v.to(device) for k, v in enc.items()}
    out = model(**enc, output_hidden_states=True, return_dict=True)
    hs = out.hidden_states  # (emb, layer1, ..., layerL)
    pos = enc["input_ids"].shape[1] - 1

    ln_f = model.transformer.ln_f
    lm_head = model.lm_head
    letter_ids = torch.tensor([letter_token_ids[L] for L in LETTERS], device=device)

    preds = []
    for h in hs:
        v = ln_f(h[:, pos, :])
        logits = lm_head(v).squeeze(0)  # [vocab]
        letter_logits = logits[letter_ids]
        preds.append(LETTERS[int(letter_logits.argmax())])
    return preds


def evaluate_layerwise(model, tok, dataset, letter_token_ids, device, max_len, desc="eval"):
    model.eval()
    probe = layerwise_letter_preds(model, tok, make_prompt(dataset[0]),
                                   letter_token_ids, device, max_len)
    n_layers = len(probe)
    correct = np.zeros(n_layers, dtype=np.int64)
    total = 0
    for ex in tqdm(dataset, desc=desc):
        gold = ex["answerKey"]
        if gold not in LETTERS:  # skip any malformed examples
            continue
        preds = layerwise_letter_preds(model, tok, make_prompt(ex),
                                       letter_token_ids, device, max_len)
        for i, p in enumerate(preds):
            correct[i] += int(p == gold)
        total += 1
    return correct / max(total, 1)


# ---------------------------------------------------------------------------
# Training dataset: label = answer-letter token only
# ---------------------------------------------------------------------------
class QASCTrainDataset(Dataset):
    """Produces input_ids = tokens(prompt + " <letter>") and labels that are
    -100 everywhere except the final letter position. GPT-2's built-in shift
    then computes loss only for predicting the letter given the full prompt.
    """

    def __init__(self, hf_ds, tok, max_len: int):
        self.tok = tok
        self.max_len = max_len
        # Pre-filter malformed examples and materialize token ids once.
        self.examples = []
        for ex in hf_ds:
            gold = ex["answerKey"]
            if gold not in LETTERS:
                continue
            prompt = make_prompt(ex)
            full = prompt + " " + gold
            prompt_ids = tok.encode(prompt, add_special_tokens=False)
            full_ids = tok.encode(full, add_special_tokens=False)
            # Sanity: full must end in the " <letter>" token.
            # If the example is too long, skip rather than risk truncating the answer.
            if len(full_ids) > max_len:
                continue
            if len(full_ids) != len(prompt_ids) + 1:
                # Unexpected tokenization (e.g. " " + "A" became 2 tokens). Skip.
                continue
            self.examples.append((full_ids, len(prompt_ids)))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ids, n_prompt = self.examples[idx]
        labels = [-100] * n_prompt + [ids[-1]]
        return {"input_ids": ids, "labels": labels}


def make_collate_fn(pad_id: int):
    def collate(batch):
        max_len = max(len(b["input_ids"]) for b in batch)
        input_ids, attn, labels = [], [], []
        for b in batch:
            ids, lbl = b["input_ids"], b["labels"]
            pad = max_len - len(ids)
            input_ids.append(ids + [pad_id] * pad)
            attn.append([1] * len(ids) + [0] * pad)
            labels.append(lbl + [-100] * pad)
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attn),
            "labels": torch.tensor(labels),
        }
    return collate


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train(model, train_loader, epochs, lr, weight_decay, warmup_steps, grad_clip, device):
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = len(train_loader) * epochs
    sched = get_linear_schedule_with_warmup(optim, warmup_steps, total_steps)

    model.train()
    for epoch in range(epochs):
        pbar = tqdm(train_loader, desc=f"epoch {epoch + 1}/{epochs}")
        running = 0.0
        for step, batch in enumerate(pbar, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optim.step()
            sched.step()
            optim.zero_grad()
            running += loss.item()
            pbar.set_postfix(loss=f"{running / step:.4f}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_curves(acc_before, acc_after, model_name, out_dir: Path):
    layers = np.arange(len(acc_before))
    plt.figure(figsize=(8, 4.5))
    plt.plot(layers, acc_before, marker="o", label="Pre fine-tune")
    if acc_after is not None:
        plt.plot(layers, acc_after, marker="s", label="Post fine-tune")
    plt.axhline(1 / 8, linestyle="--", color="gray", label="Random (1/8)")
    plt.xlabel("Layer (0 = embeddings)")
    plt.ylabel("Accuracy on QASC validation subset")
    plt.title(f"Per-layer answer accuracy via logit lens ({model_name})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(out_dir / f"per_layer_accuracy.{ext}", dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    # Data
    ds = load_dataset("allenai/qasc")
    train_split = ds["train"]
    if args.train_size > 0:
        train_split = train_split.shuffle(seed=args.seed).select(
            range(min(args.train_size, len(train_split)))
        )
        print(f"Subsampled training set to {len(train_split)} examples (sanity-check mode).")
    eval_split = ds["validation"].shuffle(seed=args.seed).select(range(args.eval_size))

    # Tokenizer + model
    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)

    letter_ids = verify_letter_tokens(tok)
    print("Letter token ids:", letter_ids)

    # ---- 1. Pre-FT probe ----
    acc_before = evaluate_layerwise(model, tok, eval_split, letter_ids,
                                    device, args.max_len, desc="pre-FT eval")
    print("Pre-FT final-layer accuracy:", round(float(acc_before[-1]), 4))
    np.save(out_dir / "acc_before.npy", acc_before)

    acc_after = None
    if not args.skip_train:
        # ---- 2. Fine-tune ----
        train_ds = QASCTrainDataset(train_split, tok, args.max_len)
        print(f"Training examples kept: {len(train_ds)} / {len(train_split)}")
        loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=make_collate_fn(tok.pad_token_id),
            num_workers=2,
        )
        train(model, loader, args.epochs, args.lr, args.weight_decay,
              args.warmup_steps, args.grad_clip, device)

        # ---- 3. Post-FT probe ----
        acc_after = evaluate_layerwise(model, tok, eval_split, letter_ids,
                                       device, args.max_len, desc="post-FT eval")
        print("Post-FT final-layer accuracy:", round(float(acc_after[-1]), 4))
        np.save(out_dir / "acc_after.npy", acc_after)

        # Save the fine-tuned weights once (per course guideline: don't spam checkpoints).
        model.save_pretrained(out_dir / "ft_model")
        tok.save_pretrained(out_dir / "ft_model")

    # ---- 4. Plot ----
    plot_curves(acc_before, acc_after, args.model_name, out_dir)
    print(f"Saved plots + arrays under {out_dir}/")


if __name__ == "__main__":
    main()
