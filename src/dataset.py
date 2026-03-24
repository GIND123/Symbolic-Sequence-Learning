"""
Dataset for Symbolic Regression
================================

Each sample pairs a set of numerical observations (x₁,...,xₖ, y) with the
prefix-tokenized symbolic expression that generated them.

Key design: each equation is repeated `repeats` times per epoch, each time
with different randomly sampled observation points + Gaussian noise. This
turns 77 unique equations into 3,850 effective training samples, preventing
overfitting.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List


class FeynmanDataset(Dataset):
    """
    Feynman dataset with data augmentation.

    Args:
        indices: Which equations (by index) to include
        dataset_dict: Output of preprocess_feynman()
        n_pts: Number of observation points per sample
        max_seq: Maximum prefix token sequence length
        max_vars: Pad input features to this many variables
        augment: Enable random subsampling + noise injection
        repeats: Each equation appears this many times per epoch
    """

    def __init__(self, indices, dataset_dict, n_pts=200, max_seq=64,
                 max_vars=10, augment=True, repeats=1):
        self.vocab = dataset_dict['vocab']
        self.n_pts = n_pts
        self.max_seq = max_seq
        self.max_vars = max_vars
        self.augment = augment
        self.repeats = repeats
        self.samples = []

        for i in indices:
            fname = dataset_dict['filenames'][i]
            tokens = dataset_dict['tokenized'][i]
            if tokens is not None and fname in dataset_dict['data_samples']:
                arr = dataset_dict['data_samples'][fname]
                if arr.ndim == 2 and arr.shape[0] >= 10:
                    self.samples.append({
                        'fname': fname, 'data': arr, 'tokens': tokens
                    })

        print(f"    Dataset: {len(self.samples)} eqs × {repeats} repeats = "
              f"{len(self.samples) * repeats} effective samples")

    def __len__(self):
        return len(self.samples) * self.repeats

    def __getitem__(self, idx):
        s = self.samples[idx % len(self.samples)]
        data = s['data']
        n_rows, n_cols = data.shape
        n_vars = n_cols - 1

        # Random subsample (different each time due to repeats)
        if self.augment:
            ix = np.random.choice(n_rows, self.n_pts,
                                  replace=(n_rows < self.n_pts))
        else:
            ix = np.arange(min(self.n_pts, n_rows))
            if len(ix) < self.n_pts:
                ix = np.pad(ix, (0, self.n_pts - len(ix)), mode='wrap')

        pts = data[ix].copy()

        # Gaussian noise on inputs (not output)
        if self.augment:
            noise = 0.01 * np.std(pts[:, :-1], axis=0, keepdims=True) + 1e-10
            pts[:, :-1] += np.random.randn(self.n_pts, n_vars) * noise

        x, y = pts[:, :-1], pts[:, -1:]

        # Pad to max_vars
        if n_vars < self.max_vars:
            x = np.concatenate(
                [x, np.zeros((self.n_pts, self.max_vars - n_vars))], axis=1
            )
        elif n_vars > self.max_vars:
            x = x[:, :self.max_vars]

        inp = np.concatenate([x, y], axis=1)

        # Z-score normalize
        mu = inp.mean(axis=0, keepdims=True)
        std = inp.std(axis=0, keepdims=True) + 1e-8
        inp = np.clip(
            np.nan_to_num((inp - mu) / std, nan=0.0, posinf=5.0, neginf=-5.0),
            -10, 10
        )

        # Encode target tokens
        ids = [self.vocab['<sos>']]
        for tok in s['tokens']:
            ids.append(self.vocab.get(tok, self.vocab['<unk>']))
        ids.append(self.vocab['<eos>'])

        if len(ids) < self.max_seq:
            ids += [0] * (self.max_seq - len(ids))
        else:
            ids = ids[:self.max_seq - 1] + [self.vocab['<eos>']]

        return {
            'input': torch.tensor(inp, dtype=torch.float32),
            'target': torch.tensor(ids, dtype=torch.long),
            'n_vars': n_vars,
            'fname': s['fname'],
        }
