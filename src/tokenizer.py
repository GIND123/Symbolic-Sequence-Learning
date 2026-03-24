"""
Task 1.1 — Preprocessing & Tokenization
========================================

Strategy: PREFIX (POLISH) NOTATION

Rationale:
  1. Unambiguous without parentheses — smaller vocab, shorter sequences
  2. 1-to-1 mapping to expression trees — critical for GP seeding (Task 2.6).
     Converting prefix → tree is O(n) with a recursive parser.
  3. Standard in SR literature (SymbolicGPT, NESYMRES, E2E-SR)
  4. Fixed-arity operators → constrained beam search decoding
  5. Left-folded binary ops → consistent binary trees for GP crossover

Vocabulary:
  - Special:    <pad>=0, <sos>=1, <eos>=2, <unk>=3
  - Operators:  add, sub, mul, div, pow, neg, sqrt, exp, log, sin, cos, ...
  - Constants:  pi, E
  - Variables:  x1..x10 (positional, mapped from named variables)
  - Numbers:    integer/float literals
"""

import re
import numpy as np
import pandas as pd
import sympy
from sympy import symbols
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations, implicit_multiplication_application
)
from collections import Counter
from typing import List, Dict, Optional


# ── Token Categories ─────────────────────────────────────────────────────────

SPECIAL_TOKENS = ['<pad>', '<sos>', '<eos>', '<unk>']

OPERATOR_TOKENS = [
    'add', 'sub', 'mul', 'div', 'pow', 'neg', 'sqrt', 'exp', 'log',
    'sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'sinh', 'cosh', 'tanh', 'abs',
]

CONSTANT_TOKENS = ['pi', 'E']
VARIABLE_TOKENS = [f'x{i}' for i in range(1, 11)]

SYMPY_TO_PREFIX = {
    'Add': 'add', 'Mul': 'mul', 'Pow': 'pow', 'exp': 'exp', 'log': 'log',
    'sqrt': 'sqrt', 'sin': 'sin', 'cos': 'cos', 'tan': 'tan',
    'asin': 'asin', 'acos': 'acos', 'atan': 'atan',
    'Abs': 'abs', 'tanh': 'tanh', 'cosh': 'cosh', 'sinh': 'sinh',
}

BINARY_OP_SET = {'add', 'sub', 'mul', 'div', 'pow'}
UNARY_OP_SET = {'neg', 'sqrt', 'exp', 'log', 'sin', 'cos', 'tan',
                'asin', 'acos', 'atan', 'sinh', 'cosh', 'tanh', 'abs'}


# ── Formula Parsing ──────────────────────────────────────────────────────────

def normalize_formula(s: str) -> str:
    """Clean formula string for sympy parsing."""
    s = s.strip().replace('^', '**')
    s = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', s)
    return s.replace('arcsin', 'asin').replace('arccos', 'acos').replace('arctan', 'atan')


def build_variable_map(row) -> Dict[str, str]:
    """Map named variables → positional tokens (x1, x2, ...).
    Handles NaN in '# variables' gracefully."""
    var_map = {}
    try:
        n_vars_raw = row['# variables']
    except (KeyError, TypeError):
        n_vars_raw = None

    if n_vars_raw is not None and not (isinstance(n_vars_raw, float) and np.isnan(n_vars_raw)):
        n_vars = int(n_vars_raw)
    else:
        n_vars = 0
        for i in range(1, 11):
            try:
                val = row[f'v{i}_name']
                if pd.notna(val) and str(val).strip():
                    n_vars = i
                else:
                    break
            except (KeyError, TypeError):
                break

    for i in range(1, n_vars + 1):
        try:
            val = row[f'v{i}_name']
            if pd.notna(val) and str(val).strip():
                var_map[str(val).strip()] = f'x{i}'
        except (KeyError, TypeError):
            pass
    return var_map


def substitute_variables(s: str, var_map: Dict[str, str]) -> str:
    """Replace named variables with positional tokens, longest-first."""
    for vname in sorted(var_map.keys(), key=len, reverse=True):
        s = re.sub(rf'\b{re.escape(vname)}\b', var_map[vname], s)
    return s


def encode_number(val: float) -> str:
    """Encode a numeric value as a token string."""
    if val == int(val) and abs(val) <= 100:
        return str(int(val))
    if abs(val - np.pi) < 1e-10:
        return 'pi'
    return f'{val:.6g}'


# ── Sympy → Prefix Conversion ───────────────────────────────────────────────

def sympy_to_prefix(expr) -> List[str]:
    """Convert sympy expression to prefix (Polish) notation token list."""
    if expr.is_Symbol:
        return [str(expr)]
    if expr.is_Number:
        return [encode_number(float(expr))]
    if expr == sympy.pi:
        return ['pi']
    if expr == sympy.E:
        return ['E']
    if expr.is_Mul and expr.args[0] == -1 and len(expr.args) == 2:
        return ['neg'] + sympy_to_prefix(expr.args[1])

    func_name = type(expr).__name__
    if func_name in SYMPY_TO_PREFIX:
        op = SYMPY_TO_PREFIX[func_name]
        args = expr.args
        if op in ('add', 'mul') and len(args) > 2:
            result = sympy_to_prefix(args[0])
            for arg in args[1:]:
                result = [op] + result + sympy_to_prefix(arg)
            return result
        elif op in ('add', 'mul') and len(args) == 2:
            return [op] + sympy_to_prefix(args[0]) + sympy_to_prefix(args[1])
        elif op == 'pow':
            base, ev = args
            if ev == sympy.Rational(1, 2):
                return ['sqrt'] + sympy_to_prefix(base)
            elif ev == -1:
                return ['div', '1'] + sympy_to_prefix(base)
            return [op] + sympy_to_prefix(base) + sympy_to_prefix(ev)
        else:
            ct = []
            for arg in args:
                ct.extend(sympy_to_prefix(arg))
            return [op] + ct

    if expr.is_Mul:
        numer, denom = expr.as_numer_denom()
        if denom != 1:
            return ['div'] + sympy_to_prefix(numer) + sympy_to_prefix(denom)
        args = expr.args
        result = sympy_to_prefix(args[0])
        for arg in args[1:]:
            result = ['mul'] + result + sympy_to_prefix(arg)
        return result

    if hasattr(expr, 'args') and len(expr.args) > 0:
        tokens = [type(expr).__name__.lower()]
        for arg in expr.args:
            tokens.extend(sympy_to_prefix(arg))
        return tokens
    return [str(expr)]


def prefix_to_infix(tokens: List[str]) -> str:
    """Convert prefix tokens back to readable infix string."""
    bops = {'add': '+', 'sub': '-', 'mul': '*', 'div': '/', 'pow': '**'}
    uops = UNARY_OP_SET
    stack = []
    for tok in reversed(tokens):
        if tok in bops:
            a = stack.pop() if stack else '?'
            b = stack.pop() if stack else '?'
            stack.append(f'({a} {bops[tok]} {b})')
        elif tok in uops:
            a = stack.pop() if stack else '?'
            stack.append(f'{tok}({a})')
        else:
            stack.append(tok)
    return stack[0] if stack else ''


# ── Vocabulary Builder ───────────────────────────────────────────────────────

def build_vocabulary(tokenized_formulas: List[Optional[List[str]]]) -> Dict[str, int]:
    """Build token → index vocabulary from tokenized formulas."""
    token_counter = Counter()
    for tokens in tokenized_formulas:
        if tokens is not None:
            token_counter.update(tokens)

    vocab = {}
    vid = 0
    for tok in SPECIAL_TOKENS + OPERATOR_TOKENS + CONSTANT_TOKENS + VARIABLE_TOKENS:
        vocab[tok] = vid
        vid += 1
    for tok, _ in token_counter.most_common():
        if tok not in vocab:
            vocab[tok] = vid
            vid += 1
    return vocab


# ── Full Pipeline ────────────────────────────────────────────────────────────

def preprocess_feynman(equations_csv: str, search_dirs: list) -> dict:
    """Full preprocessing pipeline for AI Feynman dataset."""
    import os

    df = pd.read_csv(equations_csv)
    df = df.dropna(subset=['Formula']).reset_index(drop=True)
    print(f"  {len(df)} equations with formulas")

    tokenized_formulas = []
    var_maps = []
    skipped = []

    for idx, row in df.iterrows():
        formula_str = str(row['Formula']).strip()
        var_map = build_variable_map(row)
        var_maps.append(var_map)

        try:
            norm = normalize_formula(formula_str)
            subst = substitute_variables(norm, var_map)
            ld = {f'x{i}': symbols(f'x{i}') for i in range(1, 11)}
            ld['pi'] = sympy.pi
            expr = parse_expr(
                subst, local_dict=ld,
                transformations=standard_transformations + (implicit_multiplication_application,)
            )
            tokenized_formulas.append(sympy_to_prefix(expr))
        except Exception as e:
            skipped.append((idx, str(e)[:80]))
            tokenized_formulas.append(None)

    vocab = build_vocabulary(tokenized_formulas)

    # Load numerical data
    data_samples = {}
    loaded, failed = 0, []
    for _, row in df.iterrows():
        fr = row.get('Filename', None)
        if fr is None or (isinstance(fr, float) and np.isnan(fr)):
            continue
        fname = str(fr).strip()
        for d in search_dirs:
            path = os.path.join(d, fname)
            if os.path.exists(path):
                try:
                    arr = np.loadtxt(path)
                    if arr.ndim == 2 and arr.shape[0] >= 10:
                        data_samples[fname] = arr
                        loaded += 1
                except:
                    failed.append(fname)
                break
        else:
            failed.append(fname)

    filenames = [
        str(row.get('Filename', f'eq_{i}')).strip()
        if pd.notna(row.get('Filename')) else f"eq_{i}"
        for i, row in df.iterrows()
    ]

    valid = [t for t in tokenized_formulas if t is not None]
    lens = [len(t) for t in valid]
    print(f"  Tokenized: {len(valid)}/{len(df)} | Vocab: {len(vocab)} | "
          f"Seq: {np.mean(lens):.1f}±{np.std(lens):.1f} (max {max(lens)})")
    print(f"  Data files: {loaded} loaded | {len(failed)} missing")

    return {
        'tokenized': tokenized_formulas,
        'vocab': vocab,
        'var_maps': var_maps,
        'data_samples': data_samples,
        'filenames': filenames,
        'dataframe': df,
    }
