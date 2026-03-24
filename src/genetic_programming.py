"""
Genetic Programming Seeded by Transformer
==========================================

This is the core innovation for Project 2.6:

  Standard GP:  random initial population → slow convergence
  Our approach:  transformer beam search → informed population → faster, better

Pipeline:
  1. Transformer beam search → K candidate prefix token sequences
  2. Convert prefix tokens → expression trees (Node objects)
  3. Seed GP population with these + mutated variants + random for diversity
  4. Evolve via crossover, mutation, tournament selection
  5. Return best expression
"""

import numpy as np
import random
from copy import deepcopy
from typing import List, Tuple, Dict, Optional


# ── Expression Tree ──────────────────────────────────────────────────────────

class Node:
    """Node in a symbolic expression tree."""

    def __init__(self, value, children=None):
        self.value = value
        self.children = children or []

    def is_leaf(self):
        return len(self.children) == 0

    def depth(self):
        if self.is_leaf():
            return 0
        return 1 + max(c.depth() for c in self.children)

    def size(self):
        if self.is_leaf():
            return 1
        return 1 + sum(c.size() for c in self.children)

    def __repr__(self):
        if self.is_leaf():
            return str(self.value)
        if len(self.children) == 1:
            return f"{self.value}({self.children[0]})"
        return f"({self.children[0]} {self.value} {self.children[1]})"


# ── Safe Operators ───────────────────────────────────────────────────────────

BINARY_OPS = {
    'add': lambda a, b: a + b,
    'sub': lambda a, b: a - b,
    'mul': lambda a, b: a * b,
    'div': lambda a, b: np.where(np.abs(b) > 1e-10, a / b, 1.0),
    'pow': lambda a, b: np.where(
        (a > 0) | (b == np.round(b)),
        np.power(np.abs(a) + 1e-10, np.clip(b, -5, 5)), 0.0
    ),
}

UNARY_OPS = {
    'neg': lambda a: -a,
    'sqrt': lambda a: np.sqrt(np.abs(a) + 1e-10),
    'exp': lambda a: np.exp(np.clip(a, -10, 10)),
    'log': lambda a: np.log(np.abs(a) + 1e-10),
    'sin': np.sin,
    'cos': np.cos,
    'tan': lambda a: np.tan(np.clip(a, -1.5, 1.5)),
    'abs': np.abs,
}


# ── Prefix ↔ Tree Conversion ────────────────────────────────────────────────

def prefix_to_tree(tokens: List[str], variables: List[str] = None) -> Node:
    """Convert prefix tokens → expression tree."""
    if variables is None:
        variables = [f'x{i}' for i in range(1, 11)]
    idx = [0]

    def parse():
        if idx[0] >= len(tokens):
            return Node('1')
        tok = tokens[idx[0]]
        idx[0] += 1
        if tok in ('<sos>', '<eos>', '<pad>', '<unk>'):
            return parse() if idx[0] < len(tokens) else Node('1')
        if tok in BINARY_OPS:
            return Node(tok, [parse(), parse()])
        if tok in UNARY_OPS:
            return Node(tok, [parse()])
        if tok in variables:
            return Node(tok)
        try:
            float(tok)
            return Node(tok)
        except ValueError:
            if tok == 'pi':
                return Node(str(np.pi))
            if tok == 'E':
                return Node(str(np.e))
            return Node('1')

    try:
        return parse()
    except:
        return Node('1')


# ── Tree Evaluation ──────────────────────────────────────────────────────────

def eval_tree(node: Node, X: np.ndarray,
              variables: List[str] = None) -> np.ndarray:
    """Evaluate expression tree on data."""
    if variables is None:
        variables = [f'x{i}' for i in range(1, X.shape[1] + 1)]
    v2c = {v: i for i, v in enumerate(variables)}

    def _e(n):
        if n.is_leaf():
            if n.value in v2c:
                return X[:, v2c[n.value]].copy()
            try:
                return np.full(X.shape[0], float(n.value))
            except:
                return np.ones(X.shape[0])
        if n.value in UNARY_OPS:
            return UNARY_OPS[n.value](_e(n.children[0]))
        if n.value in BINARY_OPS:
            return BINARY_OPS[n.value](_e(n.children[0]), _e(n.children[1]))
        return np.ones(X.shape[0])

    try:
        return np.nan_to_num(_e(node), nan=1e6, posinf=1e6, neginf=-1e6)
    except:
        return np.full(X.shape[0], 1e6)


# ── Fitness Functions ────────────────────────────────────────────────────────

def fitness_nmse(tree: Node, X: np.ndarray, y: np.ndarray,
                 variables: List[str] = None) -> float:
    """NMSE + parsimony pressure."""
    yp = eval_tree(tree, X, variables)
    return np.mean((y - yp) ** 2) / (np.var(y) + 1e-10) + 0.001 * tree.size()


def fitness_r2(tree: Node, X: np.ndarray, y: np.ndarray,
               variables: List[str] = None) -> float:
    """R² coefficient of determination."""
    yp = eval_tree(tree, X, variables)
    ss_res = np.sum((y - yp) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2) + 1e-10
    return 1.0 - ss_res / ss_tot


# ── GP Operators ─────────────────────────────────────────────────────────────

def random_tree(depth: int, variables: List[str], p_leaf: float = 0.3) -> Node:
    """Generate a random expression tree."""
    if depth <= 0 or (depth <= 2 and random.random() < p_leaf):
        if random.random() < 0.7:
            return Node(random.choice(variables))
        else:
            return Node(str(round(random.uniform(-5, 5), 2)))
    if random.random() < 0.3:
        op = random.choice(list(UNARY_OPS.keys()))
        return Node(op, [random_tree(depth - 1, variables)])
    op = random.choice(list(BINARY_OPS.keys()))
    return Node(op, [random_tree(depth - 1, variables),
                     random_tree(depth - 1, variables)])


def _collect_nodes(tree):
    """Collect (parent, child_index) pairs for subtree operations."""
    result = [(None, 0)]
    def _t(n):
        for i, c in enumerate(n.children):
            result.append((n, i))
            _t(c)
    _t(tree)
    return result


def crossover(p1: Node, p2: Node) -> Tuple[Node, Node]:
    """Subtree crossover between two parents."""
    c1, c2 = deepcopy(p1), deepcopy(p2)
    n1, n2 = _collect_nodes(c1), _collect_nodes(c2)
    if len(n1) < 2 or len(n2) < 2:
        return c1, c2
    pa1, i1 = random.choice(n1[1:])
    pa2, i2 = random.choice(n2[1:])
    if pa1 and pa2:
        pa1.children[i1], pa2.children[i2] = pa2.children[i2], pa1.children[i1]
    return c1, c2


def mutate(tree: Node, variables: List[str], p: float = 0.1) -> Node:
    """Point mutation on random nodes."""
    tree = deepcopy(tree)
    def _m(n):
        if random.random() < p:
            if n.is_leaf():
                n.value = (random.choice(variables) if random.random() < 0.5
                           else str(round(random.uniform(-5, 5), 2)))
            elif len(n.children) == 2:
                n.value = random.choice(list(BINARY_OPS.keys()))
            elif len(n.children) == 1:
                n.value = random.choice(list(UNARY_OPS.keys()))
        for c in n.children:
            _m(c)
    _m(tree)
    return tree


def tournament_select(pop, k=5):
    """Tournament selection."""
    t = random.sample(pop, min(k, len(pop)))
    t.sort(key=lambda x: x[0])
    return deepcopy(t[0][1])


# ── Main GP Loop ────────────────────────────────────────────────────────────

def run_gp(X: np.ndarray, y: np.ndarray, seed_trees: List[Node],
           variables: List[str], pop_size: int = 150, n_gen: int = 75,
           verbose: bool = True) -> Tuple[Node, float, dict]:
    """
    Genetic programming with optional transformer seeding.

    Args:
        X: Input features (n_samples, n_vars)
        y: Target values (n_samples,)
        seed_trees: Expression trees from transformer beam search (empty for baseline)
        variables: Variable names ['x1', 'x2', ...]
        pop_size: Population size
        n_gen: Number of generations

    Returns:
        best_tree, best_fitness, history
    """
    pop = []

    # (1) Seed from transformer candidates
    for t in seed_trees:
        if t and t.depth() <= 10:
            pop.append((fitness_nmse(t, X, y, variables), t))

    # (2) Mutated variants for exploration
    for t in seed_trees:
        if t:
            for _ in range(3):
                m = mutate(t, variables, 0.3)
                if m.depth() <= 10:
                    pop.append((fitness_nmse(m, X, y, variables), m))

    # (3) Random individuals for diversity
    n_seeded = len(pop)
    while len(pop) < pop_size:
        t = random_tree(5, variables)
        pop.append((fitness_nmse(t, X, y, variables), t))

    pop.sort(key=lambda x: x[0])
    pop = pop[:pop_size]

    if verbose:
        print(f"    Pop: {n_seeded} seeded + {pop_size - n_seeded} random | "
              f"Init NMSE: {pop[0][0]:.6f}")

    hist = {'best': [], 'r2': []}

    for gen in range(n_gen):
        # Elitism: keep top 10%
        elite_n = max(2, pop_size // 10)
        new_pop = [(f, deepcopy(t)) for f, t in pop[:elite_n]]

        while len(new_pop) < pop_size:
            if random.random() < 0.7:
                c1, c2 = crossover(
                    tournament_select(pop), tournament_select(pop)
                )
                for c in [c1, c2]:
                    if c.depth() <= 10:
                        new_pop.append((fitness_nmse(c, X, y, variables), c))
            else:
                c = mutate(tournament_select(pop), variables, 0.15)
                if c.depth() <= 10:
                    new_pop.append((fitness_nmse(c, X, y, variables), c))

        new_pop.sort(key=lambda x: x[0])
        pop = new_pop[:pop_size]

        bf = pop[0][0]
        hist['best'].append(bf)
        hist['r2'].append(fitness_r2(pop[0][1], X, y, variables))

        if verbose and (gen % 15 == 0 or gen == n_gen - 1):
            print(f"    Gen {gen:3d}: NMSE={bf:.6f}, R²={hist['r2'][-1]:.6f}")

        if bf < 1e-10:
            if verbose:
                print(f"    Perfect fit at gen {gen}!")
            break

    return pop[0][1], pop[0][0], hist
