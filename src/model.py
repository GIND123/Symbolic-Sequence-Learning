"""
Set Transformer Encoder + Autoregressive Decoder
==================================================

Architecture design:

1. SET TRANSFORMER ENCODER (ISAB):
   Input is a SET of (x, y) observation points — order shouldn't matter.
   ISAB uses induced attention for O(n·m) complexity instead of O(n²).
   PMA pools the set into a fixed number of latent tokens.

2. AUTOREGRESSIVE DECODER:
   Generates prefix-notation tokens left-to-right with causal masking.
   Cross-attends to the encoder's latent representation.

3. BEAM SEARCH:
   Top-K candidate expressions → seed GP population.

References:
  - Set Transformer (Lee et al., 2019)
  - SymbolicGPT (Li et al., 2022)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict


# ── Set Transformer Components ───────────────────────────────────────────────

class MAB(nn.Module):
    """Multihead Attention Block."""
    def __init__(self, d, h, drop=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, h, dropout=drop, batch_first=True)
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(
            nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(drop),
            nn.Linear(4 * d, d), nn.Dropout(drop),
        )

    def forward(self, Q, K):
        h = self.ln1(Q + self.attn(Q, K, K)[0])
        return self.ln2(h + self.ff(h))


class ISAB(nn.Module):
    """Induced Set Attention Block — O(n·m) complexity."""
    def __init__(self, d, h, m=32, drop=0.1):
        super().__init__()
        self.I = nn.Parameter(torch.randn(1, m, d) * 0.02)
        self.mab1 = MAB(d, h, drop)
        self.mab2 = MAB(d, h, drop)

    def forward(self, X):
        H = self.mab1(self.I.expand(X.size(0), -1, -1), X)
        return self.mab2(X, H)


# ── Encoder ──────────────────────────────────────────────────────────────────

class SetEncoder(nn.Module):
    """Set Transformer encoder: observations → latent tokens."""
    def __init__(self, input_dim=11, d=256, h=8, n_layers=3,
                 m=32, n_latent=16, drop=0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, d), nn.GELU(),
            nn.LayerNorm(d), nn.Linear(d, d),
        )
        self.layers = nn.ModuleList([ISAB(d, h, m, drop) for _ in range(n_layers)])
        self.seeds = nn.Parameter(torch.randn(1, n_latent, d) * 0.02)
        self.pool = MAB(d, h, drop)

    def forward(self, x):
        h = self.proj(x)
        for layer in self.layers:
            h = layer(h)
        return self.pool(self.seeds.expand(x.size(0), -1, -1), h)


# ── Decoder ──────────────────────────────────────────────────────────────────

class PosEnc(nn.Module):
    """Sinusoidal positional encoding."""
    def __init__(self, d, max_len=128, drop=0.1):
        super().__init__()
        self.dropout = nn.Dropout(drop)
        pe = torch.zeros(max_len, d)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class Decoder(nn.Module):
    """Autoregressive prefix-notation token decoder."""
    def __init__(self, vocab_size, d=256, h=8, n_layers=4, max_len=64, drop=0.1):
        super().__init__()
        self.d = d
        self.embed = nn.Embedding(vocab_size, d, padding_idx=0)
        self.pos = PosEnc(d, max_len, drop)
        layer = nn.TransformerDecoderLayer(
            d, h, 4 * d, drop, 'gelu', batch_first=True
        )
        self.dec = nn.TransformerDecoder(layer, n_layers)
        self.out = nn.Linear(d, vocab_size)

    def forward(self, tgt_ids, memory, mask=None):
        sl = tgt_ids.size(1)
        if mask is None:
            mask = nn.Transformer.generate_square_subsequent_mask(
                sl, device=tgt_ids.device
            )
        h = self.pos(self.embed(tgt_ids) * math.sqrt(self.d))
        h = self.dec(h, memory, tgt_mask=mask,
                     tgt_key_padding_mask=(tgt_ids == 0))
        return self.out(h)


# ── Full Model ───────────────────────────────────────────────────────────────

class SymRegTransformer(nn.Module):
    """
    Set Transformer Encoder → Autoregressive Decoder.
    Input:  Set of numerical (x, y) observation points
    Output: Prefix-notation symbolic expression
    """

    def __init__(self, vocab_size, input_dim=11, d=256, h=8, enc_layers=3,
                 dec_layers=4, m=32, n_latent=16, max_len=64, drop=0.15):
        super().__init__()
        self.encoder = SetEncoder(input_dim, d, h, enc_layers, m, n_latent, drop)
        self.decoder = Decoder(vocab_size, d, h, dec_layers, max_len, drop)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, points, tgt_ids):
        """
        points: (B, n_points, input_dim)
        tgt_ids: (B, seq_len) — teacher forcing targets
        Returns: (B, seq_len, vocab_size) logits
        """
        return self.decoder(tgt_ids, self.encoder(points))

    @torch.no_grad()
    def beam_search(self, points, sos_id, eos_id, beam_width=15, max_len=64):
        """
        Beam search — returns top-K candidate expressions.
        These candidates seed the genetic programming population.
        """
        self.eval()
        memory = self.encoder(points)
        all_results = []

        for b in range(points.size(0)):
            mem = memory[b:b + 1]
            beams = [(0.0, [sos_id])]
            completed = []

            for _ in range(max_len):
                candidates = []
                for score, seq in beams:
                    if seq[-1] == eos_id:
                        completed.append((score, seq))
                        continue
                    tgt = torch.tensor([seq], dtype=torch.long,
                                       device=points.device)
                    logits = self.decoder(tgt, mem)
                    log_probs = F.log_softmax(logits[0, -1], dim=-1)
                    topk_lp, topk_ids = torch.topk(log_probs, beam_width)
                    for lp, tid in zip(topk_lp.tolist(), topk_ids.tolist()):
                        candidates.append((score + lp, seq + [tid]))

                candidates.sort(key=lambda x: x[0], reverse=True)
                beams = candidates[:beam_width]
                if not beams:
                    break

            completed.extend(beams)
            completed.sort(key=lambda x: x[0], reverse=True)
            all_results.append([seq for _, seq in completed[:beam_width]])

        return all_results
