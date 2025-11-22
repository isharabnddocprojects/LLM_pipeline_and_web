#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from minimal_llm import MiniGPT, SimpleTokenizer


def load_texts(shard_path: Path, limit: int | None = None) -> List[str]:
    # Load text documents from JSONL file, optionally limiting count for faster evaluation
    texts: List[str] = []
    with shard_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            record = json.loads(line)
            text = record.get("text", "")
            if text:
                texts.append(text)
            if limit is not None and len(texts) >= limit:
                break
    if not texts:
        raise ValueError(f"No usable samples found in {shard_path}")
    return texts


def encode_texts(tokenizer: SimpleTokenizer, texts: Sequence[str]) -> List[List[int]]:
    # Tokenize all texts into sequences of token indices
    encoded: List[List[int]] = []
    for text in texts:
        tokens = tokenizer.encode(text)
        if tokens:
            encoded.append(tokens)
    if not encoded:
        raise ValueError("Tokenizer produced zero-length dataset.")
    return encoded


def load_checkpoint(model_dir: Path, device: torch.device):
    # Load model checkpoint, tokenizer, and config from directory
    checkpoint_path = model_dir / "model.pt"
    tokenizer_path = model_dir / "tokenizer.pkl"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    model = MiniGPT(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    tokenizer = SimpleTokenizer()
    tokenizer.load(tokenizer_path)

    return model, tokenizer, config


def average_next_n_loss(
    model: MiniGPT,
    token_sequences: Sequence[Sequence[int]],
    context_len: int,
    pred_len: int,
    device: torch.device,
    stride: int,
    max_samples_per_text: int = 3,
) -> float:
    # Compute average cross-entropy loss for next-token prediction over multiple positions
    # Uses sliding window sampling to evaluate model performance efficiently
    context_cap = min(context_len, model.config["context_length"])
    if context_cap <= 0:
        raise ValueError("Context length must be positive.")

    total_loss = 0.0
    total_positions = 0

    with torch.no_grad():
        for tokens in token_sequences:
            # Skip sequences too short for evaluation
            if len(tokens) < context_cap + pred_len:
                continue
            tensor = torch.tensor(tokens, dtype=torch.long, device=device)
            max_start = len(tokens) - (context_cap + pred_len)
            # Sample starting positions with stride to reduce computation
            start_positions = list(range(0, max_start + 1, max(1, stride)))[:max_samples_per_text]
            
            for start in start_positions:
                target_start = start + context_cap
                target_end = target_start + pred_len
                if target_end > len(tokens):
                    break
                # Sample positions within prediction window (not every position for speed)
                sample_step = max(1, pred_len // 10)
                for pos in range(target_start, target_end, sample_step):
                    # Extract context window ending at current position
                    hist_start = max(pos - context_cap, start)
                    input_seq = tensor[hist_start:pos]
                    if input_seq.numel() == 0:
                        continue
                    # Forward pass: predict next token
                    logits, _ = model(input_seq.unsqueeze(0))
                    next_logits = logits[:, -1, :]
                    target_token = tensor[pos].view(1)
                    # Compute cross-entropy loss for this prediction
                    loss = F.cross_entropy(
                        next_logits, target_token.to(device), reduction="sum"
                    )
                    total_loss += loss.item()
                    total_positions += 1
    if total_positions == 0:
        return math.nan
    return total_loss / total_positions


def evaluate_models(args: argparse.Namespace):
    # Main evaluation function: compare two models on held-out test data
    device = torch.device(args.device)
    shard_path = Path(args.dataset)
    texts = load_texts(shard_path, args.max_texts)

    # Load both model checkpoints
    new_model, new_tokenizer, new_config = load_checkpoint(Path(args.new_model_dir), device)
    old_model, old_tokenizer, old_config = load_checkpoint(Path(args.old_model_dir), device)

    # Tokenize texts with respective tokenizers
    new_sequences = encode_texts(new_tokenizer, texts)
    old_sequences = encode_texts(old_tokenizer, texts)

    new_context = new_config["context_length"]
    old_context = old_config["context_length"]

    # Evaluate at multiple prediction lengths to see how loss scales
    prediction_lengths = list(range(args.start_tokens, args.max_tokens + 1, args.step))
    new_scores = []
    old_scores = []

    print(f"Evaluating models on {len(texts)} texts...")
    print(f"Prediction lengths: {prediction_lengths}")
    print(f"Stride: {args.stride}, Max context: {min(new_context, args.max_context)}\n")
    print(f"{'N tokens':>10} | {'new_loss':>12} | {'old_loss':>12}")
    print("-" * 39)

    # Evaluate each model at each prediction length
    for idx, n in enumerate(prediction_lengths):
        print(f"Evaluating N={n} ({idx+1}/{len(prediction_lengths)})...", end=" ", flush=True)
        new_loss = average_next_n_loss(
            new_model,
            new_sequences,
            min(new_context, args.max_context),
            n,
            device,
            stride=args.stride,
        )
        old_loss = average_next_n_loss(
            old_model,
            old_sequences,
            min(old_context, args.max_context),
            n,
            device,
            stride=args.stride,
        )
        new_scores.append(new_loss)
        old_scores.append(old_loss)
        print(f"Done: {new_loss:12.4f} | {old_loss:12.4f}")

    # Generate comparison plot
    plot_path = Path(args.plot_path)
    plt.figure(figsize=(8, 5))
    plt.plot(prediction_lengths, new_scores, marker="o", label=args.new_label)
    plt.plot(prediction_lengths, old_scores, marker="o", label=args.old_label)
    plt.xlabel("Predicted tokens (N)")
    plt.ylabel("Mean cross entropy (lower is better)")
    plt.title("LLM next-N token evaluation")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"\nSaved plot to {plot_path.resolve()}")


def build_parser() -> argparse.ArgumentParser:
    # Configure command-line argument parser for evaluation script
    parser = argparse.ArgumentParser(description="Evaluate two MiniGPT checkpoints.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="Test_data/shard_00005.jsonl",
        help="Path to JSONL shard containing reference texts.",
    )
    parser.add_argument(
        "--new-model-dir",
        type=str,
        default="models",
        help="Directory containing the newer model checkpoint/tokenizer.",
    )
    parser.add_argument(
        "--old-model-dir",
        type=str,
        default="models_old",
        help="Directory containing the older model checkpoint/tokenizer.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device to use (e.g., cpu or cuda).",
    )
    parser.add_argument(
        "--max-texts",
        type=int,
        default=20,
        help="Limit number of shard entries to speed up evaluation.",
    )
    parser.add_argument(
        "--start-tokens",
        type=int,
        default=50,
        help="Number of tokens for the first evaluation round.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=500,
        help="Maximum number of tokens to predict.",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=50,
        help="Increment for the prediction length.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=100,
        help="Stride for sliding window over tokenized shards (larger = faster).",
    )
    parser.add_argument(
        "--plot-path",
        type=str,
        default="llm_evaluation.png",
        help="Output path for the evaluation plot.",
    )
    parser.add_argument(
        "--max-context",
        type=int,
        default=256,
        help="Cap on context length to use when feeding the models (smaller = faster).",
    )
    parser.add_argument("--new-label", type=str, default="New model", help="Plot label.")
    parser.add_argument("--old-label", type=str, default="Old model", help="Plot label.")
    return parser


if __name__ == "__main__":
    parser = build_parser()
    evaluate_models(parser.parse_args())
