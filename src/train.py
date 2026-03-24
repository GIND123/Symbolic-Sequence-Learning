"""
Training & Evaluation Pipeline
===============================

1. Preprocess AI Feynman dataset (Task 1.1)
2. Train Set Transformer with data augmentation
3. Evaluate: Transformer-only vs GP-only vs Transformer+GP (Task 2.6)
4. Print ablation metrics

Usage:
    python train.py --csv FeynmanEquations.csv --data_dir Feynman_with_units
"""

import os
import sys
import math
import json
import time
import random
import argparse
import warnings

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from tokenizer import preprocess_feynman, prefix_to_infix
from dataset import FeynmanDataset
from model import SymRegTransformer
from genetic_programming import (
    prefix_to_tree, fitness_nmse, fitness_r2, run_gp
)

warnings.filterwarnings('ignore')


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(model, train_loader, val_loader, vocab_size, config, device):
    """Train transformer with warmup + cosine schedule."""
    optimizer = optim.AdamW(model.parameters(), lr=config.lr,
                            weight_decay=config.weight_decay)

    def lr_lambda(epoch):
        if epoch < config.warmup_epochs:
            return (epoch + 1) / config.warmup_epochs
        progress = (epoch - config.warmup_epochs) / max(
            1, config.epochs - config.warmup_epochs
        )
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = nn.CrossEntropyLoss(ignore_index=0,
                                     label_smoothing=config.label_smoothing)

    use_amp = (device.type == 'cuda')
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    best_val_loss = float('inf')
    patience_ctr = 0
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    t0 = time.time()

    for epoch in range(config.epochs):
        # ── Train ──
        model.train()
        total_loss, nb = 0.0, 0
        for batch in train_loader:
            inp = batch['input'].to(device, non_blocking=True)
            tgt = batch['target'].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with torch.amp.autocast('cuda'):
                    logits = model(inp, tgt[:, :-1])
                    loss = criterion(logits.reshape(-1, vocab_size),
                                     tgt[:, 1:].reshape(-1))
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(inp, tgt[:, :-1])
                loss = criterion(logits.reshape(-1, vocab_size),
                                 tgt[:, 1:].reshape(-1))
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item()
            nb += 1

        scheduler.step()
        tl = total_loss / max(nb, 1)

        # ── Validate ──
        model.eval()
        vl_sum, vc, vt, vnb = 0.0, 0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                inp = batch['input'].to(device, non_blocking=True)
                tgt = batch['target'].to(device, non_blocking=True)
                if use_amp:
                    with torch.amp.autocast('cuda'):
                        logits = model(inp, tgt[:, :-1])
                        vl_sum += criterion(
                            logits.reshape(-1, vocab_size),
                            tgt[:, 1:].reshape(-1)
                        ).item()
                else:
                    logits = model(inp, tgt[:, :-1])
                    vl_sum += criterion(
                        logits.reshape(-1, vocab_size),
                        tgt[:, 1:].reshape(-1)
                    ).item()
                vnb += 1
                preds = logits.argmax(-1)
                mask = tgt[:, 1:] != 0
                vc += (preds == tgt[:, 1:]).masked_select(mask).sum().item()
                vt += mask.sum().item()

        vl = vl_sum / max(vnb, 1)
        va = vc / max(vt, 1)
        history['train_loss'].append(tl)
        history['val_loss'].append(vl)
        history['val_acc'].append(va)

        if epoch % 10 == 0 or epoch == config.epochs - 1:
            elapsed = time.time() - t0
            print(f"Ep {epoch:3d} | train={tl:.4f} val={vl:.4f} acc={va:.4f} "
                  f"lr={optimizer.param_groups[0]['lr']:.1e} | {elapsed:.0f}s")

        if vl < best_val_loss:
            best_val_loss = vl
            patience_ctr = 0
            torch.save(model.state_dict(), 'best_model.pt')
        else:
            patience_ctr += 1
            if patience_ctr >= config.patience:
                print(f"Early stop at epoch {epoch}")
                break

    model.load_state_dict(torch.load('best_model.pt', weights_only=True))
    print(f"Best val loss: {best_val_loss:.4f} | Time: {time.time() - t0:.0f}s")
    return history


def evaluate(model, test_ds, dataset, device, config):
    """Full evaluation: Transformer vs GP-only vs Transformer+GP."""
    inv_vocab = {v: k for k, v in dataset['vocab'].items()}
    sos_id = dataset['vocab']['<sos>']
    eos_id = dataset['vocab']['<eos>']

    results = {'transformer': [], 'transformer_gp': [], 'gp_only': []}
    model.eval()

    for i in range(len(test_ds)):
        s = test_ds[i]
        fname = s['fname']
        raw = dataset['data_samples'].get(fname)
        if raw is None:
            continue

        X, y = raw[:, :-1], raw[:, -1]
        nv = X.shape[1]
        variables = [f'x{j}' for j in range(1, nv + 1)]

        # Subsample for GP speed
        if X.shape[0] > 10000:
            gi = np.random.choice(X.shape[0], 10000, replace=False)
            Xg, yg = X[gi], y[gi]
        else:
            Xg, yg = X, y

        print(f"\n{'─' * 50}")
        print(f"[{i + 1}/{len(test_ds)}] {fname} ({nv} vars, {X.shape[0]} pts)")

        inp = s['input'].unsqueeze(0).to(device)

        # Beam search
        beams = model.beam_search(inp, sos_id, eos_id,
                                  config.beam_width, config.max_seq)
        trees = []
        for seq in beams[0]:
            toks = [inv_vocab.get(t, '<unk>') for t in seq
                    if t not in (sos_id, eos_id, 0)]
            tree = prefix_to_tree(toks, variables)
            if tree:
                trees.append(tree)

        # (1) Transformer-only
        if trees:
            scores = sorted(
                [(fitness_nmse(t, Xg, yg, variables), t) for t in trees],
                key=lambda x: x[0]
            )
            t_nmse = scores[0][0]
            t_r2 = fitness_r2(scores[0][1], Xg, yg, variables)
        else:
            t_nmse, t_r2 = 999.0, -999.0
        print(f"  Transformer:    NMSE={t_nmse:.6f}  R²={t_r2:.6f}")
        results['transformer'].append({
            'fname': fname, 'nmse': t_nmse, 'r2': t_r2
        })

        # (2) Transformer + GP
        print(f"  Transformer+GP:")
        bt, bf, hist = run_gp(Xg, yg, trees, variables,
                              config.gp_pop, config.gp_gens)
        tgp_r2 = fitness_r2(bt, Xg, yg, variables)
        results['transformer_gp'].append({
            'fname': fname, 'nmse': bf, 'r2': tgp_r2,
            'expr': str(bt)[:120], 'hist': hist,
        })

        # (3) GP-only baseline
        print(f"  GP-only:")
        bt2, bf2, _ = run_gp(Xg, yg, [], variables,
                              config.gp_pop, config.gp_gens)
        gp_r2 = fitness_r2(bt2, Xg, yg, variables)
        results['gp_only'].append({
            'fname': fname, 'nmse': bf2, 'r2': gp_r2
        })

    return results


def print_summary(results):
    """Print final evaluation metrics."""
    print("\n" + "=" * 70)
    print("FINAL METRICS")
    print("=" * 70)

    for method, label in [
        ('transformer', 'Transformer Only'),
        ('gp_only', 'GP Only (Random Init)'),
        ('transformer_gp', 'Transformer + GP (Ours)'),
    ]:
        entries = results[method]
        if not entries:
            continue
        nmses = [e['nmse'] for e in entries]
        r2s = [max(e['r2'], -1.0) for e in entries]
        print(f"\n  {label}:")
        print(f"    Mean NMSE:   {np.mean(nmses):.6f} ± {np.std(nmses):.6f}")
        print(f"    Median NMSE: {np.median(nmses):.6f}")
        print(f"    Mean R²:     {np.mean(r2s):.6f} ± {np.std(r2s):.6f}")
        print(f"    Median R²:   {np.median(r2s):.6f}")
        print(f"    R² > 0.9:    {sum(1 for r in r2s if r > 0.9)}/{len(entries)}")
        print(f"    R² > 0.99:   {sum(1 for r in r2s if r > 0.99)}/{len(entries)}")
        print(f"    NMSE < 0.01: {sum(1 for n in nmses if n < 0.01)}/{len(entries)}")

    if results['transformer_gp'] and results['gp_only']:
        imps = [
            (g['nmse'] - t['nmse']) / g['nmse'] * 100
            for t, g in zip(results['transformer_gp'], results['gp_only'])
            if g['nmse'] > 1e-10
        ]
        if imps:
            print(f"\n  ABLATION — Seeding Benefit:")
            print(f"    Mean improvement:   {np.mean(imps):.1f}%")
            print(f"    Median improvement: {np.median(imps):.1f}%")
            print(f"    Helped: {sum(1 for x in imps if x > 0)}/{len(imps)}")


def save_plots(history, results, path='results_plot.png'):
    """Save training and evaluation plots."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Training curves
    axes[0].plot(history['train_loss'], label='Train', lw=2)
    axes[0].plot(history['val_loss'], label='Val', lw=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].set_yscale('log')
    axes[0].grid(True, alpha=0.3)

    # R² comparison
    labels = ['Transformer', 'GP Only', 'Trans+GP']
    r2m = [
        np.mean([max(e['r2'], -1) for e in results[k]]) if results[k] else 0
        for k in ['transformer', 'gp_only', 'transformer_gp']
    ]
    colors = ['#4C72B0', '#DD8452', '#55A868']
    bars = axes[1].bar(labels, r2m, color=colors, edgecolor='black', lw=0.5)
    axes[1].set_ylabel('Mean R²')
    axes[1].set_title('Method Comparison')
    axes[1].set_ylim([-0.5, 1.0])
    axes[1].axhline(0, color='gray', ls='--', alpha=0.5)
    axes[1].grid(True, alpha=0.3, axis='y')
    for b, v in zip(bars, r2m):
        axes[1].text(b.get_x() + b.get_width() / 2, max(b.get_height(), 0) + 0.02,
                     f'{v:.3f}', ha='center', fontweight='bold')

    # GP convergence
    sc = [e['hist']['best'] for e in results['transformer_gp'] if 'hist' in e]
    if sc:
        ml = max(len(c) for c in sc)
        padded = np.array([c + [c[-1]] * (ml - len(c)) for c in sc])
        axes[2].plot(np.mean(padded, axis=0), label='Seeded GP',
                     color='#55A868', lw=2)
        axes[2].set_xlabel('Generation')
        axes[2].set_ylabel('Best NMSE')
        axes[2].set_title('GP Convergence')
        axes[2].legend()
        axes[2].set_yscale('log')
        axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {path}")


# ── Main ─────────────────────────────────────────────────────────────────────

class Config:
    # Data
    max_vars = 10
    n_points = 200
    max_seq = 64
    train_repeats = 50

    # Model
    d_model = 256
    n_heads = 8
    enc_layers = 3
    dec_layers = 4
    dropout = 0.15

    # Training
    batch_size = 16
    lr = 3e-4
    weight_decay = 1e-2
    epochs = 300
    patience = 40
    warmup_epochs = 10
    label_smoothing = 0.1

    # Evaluation
    beam_width = 15
    gp_pop = 150
    gp_gens = 75

    seed = 42


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default='FeynmanEquations.csv')
    parser.add_argument('--data_dir', default='Feynman_with_units')
    args = parser.parse_args()

    config = Config()
    set_seed(config.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {torch.cuda.get_device_name(0)} | "
              f"{props.total_memory / 1e9:.1f} GB")

    # ── Preprocessing ──
    print("\n" + "=" * 60)
    print("TASK 1.1 — PREPROCESSING")
    print("=" * 60)

    search_dirs = [args.data_dir]
    if os.path.isdir(args.data_dir):
        for sub in os.listdir(args.data_dir):
            sp = os.path.join(args.data_dir, sub)
            if os.path.isdir(sp):
                search_dirs.append(sp)

    dataset = preprocess_feynman(args.csv, search_dirs)

    # Show examples
    print("\n--- Examples ---")
    ct = 0
    for i in range(len(dataset['tokenized'])):
        t = dataset['tokenized'][i]
        if t is not None:
            f = str(dataset['dataframe'].iloc[i]['Formula'])
            print(f"  [{i}] {f}")
            print(f"       Prefix: {t}")
            print(f"       Infix:  {prefix_to_infix(t)}")
            ct += 1
            if ct >= 3:
                break

    # ── Data Splits ──
    print("\n" + "=" * 60)
    print("DATA SPLITS (80-10-10)")
    print("=" * 60)

    valid_idx = [
        i for i in range(len(dataset['filenames']))
        if dataset['tokenized'][i] is not None
        and dataset['filenames'][i] in dataset['data_samples']
    ]
    rng = np.random.RandomState(config.seed)
    rng.shuffle(valid_idx)
    n = len(valid_idx)
    train_idx = valid_idx[:int(0.8 * n)]
    val_idx = valid_idx[int(0.8 * n):int(0.9 * n)]
    test_idx = valid_idx[int(0.9 * n):]
    print(f"  train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    train_ds = FeynmanDataset(train_idx, dataset, config.n_points,
                              config.max_seq, config.max_vars,
                              augment=True, repeats=config.train_repeats)
    val_ds = FeynmanDataset(val_idx, dataset, config.n_points,
                            config.max_seq, config.max_vars,
                            augment=False, repeats=1)
    test_ds = FeynmanDataset(test_idx, dataset, config.n_points,
                             config.max_seq, config.max_vars,
                             augment=False, repeats=1)

    is_cuda = (device.type == 'cuda')
    train_loader = DataLoader(train_ds, config.batch_size, shuffle=True,
                              drop_last=True, num_workers=2, pin_memory=is_cuda)
    val_loader = DataLoader(val_ds, config.batch_size, shuffle=False,
                            num_workers=2, pin_memory=is_cuda)

    # ── Model ──
    vocab_size = len(dataset['vocab'])
    model = SymRegTransformer(
        vocab_size, config.max_vars + 1, config.d_model, config.n_heads,
        config.enc_layers, config.dec_layers, 32, 16, config.max_seq,
        config.dropout,
    ).to(device)
    print(f"\nModel: {sum(p.numel() for p in model.parameters()):,} params | "
          f"Vocab: {vocab_size}")

    # ── Train ──
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)
    history = train(model, train_loader, val_loader, vocab_size, config, device)

    # ── Evaluate ──
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)
    results = evaluate(model, test_ds, dataset, device, config)

    # ── Summary ──
    print_summary(results)

    # ── Save ──
    save_plots(history, results, 'results/results_plot.png')

    with open('results/results.json', 'w') as f:
        save_data = {
            k: [{kk: vv for kk, vv in e.items() if kk != 'hist'} for e in v]
            for k, v in results.items()
        }
        json.dump(save_data, f, indent=2, default=str)
    print("Results saved to results/results.json")

    print("\n" + "=" * 60)
    print("✓ DONE")
    print("=" * 60)


if __name__ == '__main__':
    main()
