#!/usr/bin/env python3
"""
saito_loop_desktop_full.py
================================================================================
The Saito Loop - ONE self-contained desktop file. No project imports.

Everything the paper needs is inlined here:
  PART A  identifiability -> DAG -> unique order (Thm 2); information-limited
          recovery -> comparison -> unique milestone order; 729-point robustness
          grid; non-uniform-onset falsifiability (Thm 3); competing models;
          Figures 1-4 (Figure 4 is the revised two-tier model-space audit:
          top row = the three competing accounts on uniqueness and
          dissociation; bottom row = 729 combinations -> one and the same
          order, shown as robustness).
  PART B  reference dynamical audit of the four (P,S) families vs three
          prespecified signatures.
  PART C  formal bounded classification: one-edit N1(M) and Boolean (P,S).

External libraries (desktop only): numpy, networkx, matplotlib.
This file does NOT run in Pythonista; for iOS use saito_loop_pythonista.py.

Run:
    python saito_loop_desktop_full.py            # report + figures here
    python saito_loop_desktop_full.py ./out      # figures written to ./out
================================================================================
"""
from __future__ import annotations

# =============================================================================
# NOTE ON WHAT THE NUMERICAL PART ESTABLISHES  (read this before the code)
# =============================================================================
# This script is a CONSISTENCY CHECK on the analytical derivation, not the
# derivation itself. The prerequisite edges
#
#     pi_s -> beta -> pi_m -> pi_vfast -> pi_vslow
#
# are supplied to the code as the support of the generative gating (see
# PREREQUISITE_EDGES / the MODEL section below). Stage 1 then confirms that this
# encoded structure reproduces the same dependency graph and the same unique
# topological order that Theorem 2 derives ANALYTICALLY. What forces the edges is
# the likelihood-flattening argument of Theorem 2 -- which capacity's data go
# silent without which -- an analytical result, NOT this simulation. The code
# therefore demonstrates that the derived wiring is self-consistent and yields a
# single admissible order; it does NOT discover the graph by search. Accordingly
# the deductive conclusions are independent of simulation length, random seed,
# grid resolution and context number. See Methods (Theorem 2) and the manuscript
# subsection "Role of numerical analyses".
# =============================================================================


# ============================ PART A ============================
"""saito_loop_all.py
================================================================================
The Saito Loop — a deductive theory of hierarchical recovery in catatonia.
Single-file reproduction of the entire analysis.

Run:
    python saito_loop_all.py                 # report + figures in current dir
    python saito_loop_all.py  /output/dir    # figures written to /output/dir

The script executes a two-stage deduction and prints a verification report:

  Stage 1  identifiability -> DAG -> unique topological order  (== Saito Loop)
  Stage 2  information-limited recovery law -> comparison theorem
           -> unique milestone order (== Saito Loop)

plus a 729-point robustness grid, a non-uniform-onset analysis, a comparison
with competing models, and Figures 1-4.

Dependencies: numpy, scipy(optional), matplotlib, networkx.

Sections below correspond one-to-one to the modules model / identifiability /
topology / recovery / theorem / competing_models / figures / reproduce.
================================================================================
"""

import os
import sys
import json
import math
import itertools

import numpy as np
import networkx as nx


def _get_ipython():
    try:
        from IPython import get_ipython
        return get_ipython()
    except Exception:
        return None


def _in_notebook():
    """True only when executing inside a live IPython/Jupyter/Colab kernel.
    (A `!python script.py` subprocess is NOT a notebook: it returns False.)"""
    if "google.colab" in sys.modules:
        return True
    ip = _get_ipython()
    return ip is not None and ip.__class__.__name__ in ("ZMQInteractiveShell", "Shell")


IN_NOTEBOOK = _in_notebook()

import matplotlib
if IN_NOTEBOOK:
    # Force the inline backend even if a previous cell (e.g. the original
    # script) already switched matplotlib to the non-drawing "Agg" backend,
    # which stays active for the whole kernel session and silently blocks
    # inline figures. This is the usual reason plots "don't render".
    ip = _get_ipython()
    if ip is not None:
        try:
            ip.run_line_magic("matplotlib", "inline")
        except Exception:
            pass
else:
    matplotlib.use("Agg")  # headless script use: save to files only
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch


def _resolve_out_dir():
    """Where figures/report are written. In a notebook sys.argv carries the
    kernel's own flags (e.g. '-f /path/kernel.json'), so argv[1] must NOT be
    read as a directory there."""
    if IN_NOTEBOOK:
        return "."
    return sys.argv[1] if len(sys.argv) > 1 else "."


def _finish(fig, path):
    """Save a figure and, inside a notebook, render it inline right away."""
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    if IN_NOTEBOOK:
        try:
            from IPython.display import display
            display(fig)          # draw the figure object directly (reliable)
        except Exception:
            plt.show()
    plt.close(fig)


# =============================================================================
# 1. MODEL — the generative architecture
# =============================================================================
# Five computational quantities. Each is given a distinct computational role;
# no numerical recovery ranking, milestone ordering or temporal constraint is
# written into the architecture. The only structural commitment is a set of
# prerequisite (direct-dependency) edges implied by the information geometry:
#
#     pi_s  ->  beta  ->  pi_m  ->  pi_vfast  ->  pi_vslow
#
#   pi_s     : sensory precision
#   beta     : policy precision
#   pi_m     : motivational precision
#   pi_vfast : fast contextual volatility
#   pi_vslow : slow structural volatility  (defined over the *distribution* of
#              context-specific fast-volatility estimates, hence structurally
#              above pi_vfast)

QUANTITIES = ["pi_s", "beta", "pi_m", "pi_vfast", "pi_vslow"]

LABELS = {
    "pi_s": r"$\pi_s$",
    "beta": r"$\beta$",
    "pi_m": r"$\pi_m$",
    "pi_vfast": r"$\pi_{v\_\mathrm{fast}}$",
    "pi_vslow": r"$\pi_{v\_\mathrm{slow}}$",
}

# Direct prerequisite edges (parent -> child). The ONLY structural facts
# supplied. The clinically observed recovery order is never entered here.
PREREQUISITE_EDGES = [
    ("pi_s", "beta"),
    ("beta", "pi_m"),
    ("pi_m", "pi_vfast"),
    ("pi_vfast", "pi_vslow"),
]

# The clinically observed Saito Loop, held out for validation only. It is NOT
# used anywhere in Stage 1 or Stage 2; it is compared against the derived order
# at the very end.
CLINICAL_SAITO_LOOP = ("pi_s", "beta", "pi_m", "pi_vfast", "pi_vslow")

N_PERMUTATIONS = math.factorial(len(QUANTITIES))  # 120


def index(name):
    return QUANTITIES.index(name)


# =============================================================================
# 2. IDENTIFIABILITY — Stage 1a: zeroing-based dependency recovery
# =============================================================================
# Method: zeroing-based identifiability. Each quantity is probed by asking
# whether its likelihood remains informative after selectively zeroing
# information elsewhere. "Informative" is formalised as strictly positive
# Fisher information. We build a minimal generative model in which the Fisher
# information available for quantity k is gated multiplicatively by the
# availability of its prerequisites:
#
#     I_k(level) = base_k * prod_{p in ancestors(k)} level_p
#
# so zeroing any ancestor drives I_k to zero, while zeroing a non-ancestor
# leaves I_k unchanged. Reading off, for every ordered pair (i, k), whether
# zeroing i collapses the information for k recovers the reachability i ->* k.
# No temporal information is used at any point.

def ancestors():
    """Transitive ancestors implied by the prerequisite edges (the ground-truth
    generative wiring the identifiability probe must rediscover)."""
    parents = {q: set() for q in QUANTITIES}
    for a, b in PREREQUISITE_EDGES:
        parents[b].add(a)
    anc = {q: set() for q in QUANTITIES}
    changed = True
    while changed:                       # fixed-point transitive closure
        changed = False
        for q in QUANTITIES:
            new = set(parents[q])
            for p in list(parents[q]):
                new |= anc[p]
            if new != anc[q]:
                anc[q] = new
                changed = True
    return anc


def fisher_information(levels, base=None):
    """Per-quantity Fisher information given availability levels in [0, 1].
    Information for k is gated by the product of its ancestors' levels."""
    anc = ancestors()
    if base is None:
        base = {q: 1.0 for q in QUANTITIES}
    info = {}
    for k in QUANTITIES:
        g = 1.0
        for p in anc[k]:
            g *= levels[p]
        info[k] = base[k] * g
    return info


def zeroing_dependency_matrix(eps=1e-9):
    """Reconstruct reachability R where R[i, k] = 1 iff zeroing quantity i
    renders quantity k's likelihood non-informative. Returns (R, order)."""
    qs = QUANTITIES
    n = len(qs)
    R = np.zeros((n, n), dtype=int)
    base_levels = {q: 1.0 for q in qs}
    base_info = fisher_information(base_levels)
    for i, qi in enumerate(qs):
        levels = dict(base_levels)
        levels[qi] = 0.0                 # zeroing intervention on qi
        info = fisher_information(levels)
        for k, qk in enumerate(qs):
            if qk == qi:
                continue
            if base_info[qk] > eps and info[qk] <= eps:
                R[i, k] = 1
    return R, list(qs)


# =============================================================================
# 3. TOPOLOGY — Stage 1b: transitive reduction + unique topological order
# =============================================================================
# A DAG has a unique topological ordering iff that ordering is a total order,
# i.e. the transitive reduction is a Hamiltonian path. We verify uniqueness
# directly by enumerating topological sorts.

def dag_from_reachability(R, order):
    """Build a DiGraph from a reachability matrix and take its transitive
    reduction (removes shortcut edges, leaving direct prerequisites)."""
    G_reach = nx.DiGraph()
    G_reach.add_nodes_from(order)
    n = len(order)
    for i in range(n):
        for k in range(n):
            if R[i, k]:
                G_reach.add_edge(order[i], order[k])
    if not nx.is_directed_acyclic_graph(G_reach):
        raise ValueError("Recovered relation is not acyclic.")
    return nx.transitive_reduction(G_reach)


def unique_topological_order(G):
    """Return (order, is_unique); is_unique iff the DAG admits exactly one
    topological ordering."""
    sorts = []
    for s in nx.all_topological_sorts(G):
        sorts.append(tuple(s))
        if len(sorts) > 1:
            break
    is_unique = len(sorts) == 1
    order = next(iter(nx.all_topological_sorts(G)))
    return tuple(order), is_unique


def derived_order():
    """End-to-end Stage 1: identifiability -> DAG -> unique topological order."""
    R, order = zeroing_dependency_matrix()
    G = dag_from_reachability(R, order)
    top, is_unique = unique_topological_order(G)
    return {"reachability": R, "graph": G, "order": top, "is_unique": is_unique}


# =============================================================================
# 4. RECOVERY — Stage 2a: the information-limited recovery law
# =============================================================================
# Each quantity k has a recovery state r_k(t) in [0, 1]. The recovery law is:
#   1. information-limited with multiplicative availability
#        A_k(t) = prod_{p in ancestors(k)} r_p(t)      (A_root = 1)
#   2. preserved channel identity (own state and rate), and
#   3. monotonic improvement
#        dr_k/dt = c_k * A_k(t) * (1 - r_k(t)) >= 0.
# Because ancestor-sets are nested along the chain, availability is
# monotonically non-increasing across the hierarchy -- exactly what the
# comparison theorem needs.

def simulate(rates, onset, t_max=60.0, dt=0.02):
    """Integrate the recovery ODE with explicit RK4. rates/onset are dicts
    name -> value. Returns (T, S) with S shape (steps+1, 5)."""
    anc = ancestors()
    qs = QUANTITIES
    r = np.array([onset[q] for q in qs], dtype=float)
    c = np.array([rates[q] for q in qs], dtype=float)
    anc_idx = [[index(p) for p in anc[q]] for q in qs]

    def deriv(state):
        avail = np.ones_like(state)
        for k, parents in enumerate(anc_idx):
            for p in parents:
                avail[k] *= state[p]
        return c * avail * (1.0 - state)

    n_steps = int(round(t_max / dt))
    T = np.zeros(n_steps + 1)
    S = np.zeros((n_steps + 1, len(qs)))
    S[0] = r
    for i in range(n_steps):
        k1 = deriv(r)
        k2 = deriv(np.clip(r + 0.5 * dt * k1, 0, 1))
        k3 = deriv(np.clip(r + 0.5 * dt * k2, 0, 1))
        k4 = deriv(np.clip(r + dt * k3, 0, 1))
        r = np.clip(r + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4), 0, 1)
        S[i + 1] = r
        T[i + 1] = (i + 1) * dt
    return T, S


def milestone_times(T, S, theta=0.5):
    """First time each channel crosses threshold theta (inf if never)."""
    n = S.shape[1]
    times = np.full(n, np.inf)
    for k in range(n):
        idx = np.argmax(S[:, k] >= theta)
        if S[idx, k] >= theta:
            times[k] = T[idx]
    return times


def milestone_order(T, S, theta=0.5):
    """Permutation of quantity names ranked by milestone time."""
    times = milestone_times(T, S, theta)
    order = [QUANTITIES[i] for i in np.argsort(times, kind="stable")]
    return tuple(order), times


def milestone_times_batch(rates_arr, onset_arr, theta=0.5, t_max=80.0, dt=0.05):
    """Vectorised milestone times for a batch of configs. rates_arr/onset_arr
    are (N, 5). theta scalar or length-N. Returns (N, 5) with inf where unmet."""
    anc = ancestors()
    qs = QUANTITIES
    anc_idx = [[index(p) for p in anc[q]] for q in qs]
    r = np.array(onset_arr, dtype=float).copy()
    c = np.array(rates_arr, dtype=float)
    N = r.shape[0]
    theta = np.broadcast_to(np.asarray(theta, float), (N,)).reshape(N, 1)
    times = np.full((N, len(qs)), np.inf)

    def deriv(state):
        avail = np.ones_like(state)
        for k, parents in enumerate(anc_idx):
            for p in parents:
                avail[:, k] *= state[:, p]
        return c * avail * (1.0 - state)

    n_steps = int(round(t_max / dt))
    reached = np.zeros((N, len(qs)), dtype=bool)
    for i in range(n_steps):
        k1 = deriv(r)
        k2 = deriv(np.clip(r + 0.5 * dt * k1, 0, 1))
        k3 = deriv(np.clip(r + 0.5 * dt * k2, 0, 1))
        k4 = deriv(np.clip(r + dt * k3, 0, 1))
        r = np.clip(r + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4), 0, 1)
        newly = (~reached) & (r >= theta)
        times[newly] = (i + 1) * dt
        reached |= newly
        if reached.all():
            break
    return times


def order_from_times(times):
    """Given milestone_times (N, 5), return list of ordering tuples."""
    idx = np.argsort(times, axis=1, kind="stable")
    return [tuple(QUANTITIES[j] for j in row) for row in idx]


# =============================================================================
# 5. THEOREM — Stage 2b: comparison theorem + robustness
# =============================================================================

def no_crossings(S, tol=1e-9):
    """Under uniform onset the ordered trajectories must never cross."""
    order_idx = [index(q) for q in CLINICAL_SAITO_LOOP]
    ordered = S[:, order_idx]
    diffs = ordered[:, :-1] - ordered[:, 1:]
    return bool(np.all(diffs >= -tol))


def robustness_grid():
    """Pre-declared robustness grid over the six NUISANCE hyperparameters the
    deduction is claimed invariant to, in the information-limited regime.
    6 factors x 3 levels = 3**6 = 729 points:
        1. global rate scale g in {0.6, 1.0, 1.6}
        2. threshold theta    in {0.3, 0.5, 0.7}
        3. onset value r0      in {0.01, 0.02, 0.05}
        4. integration step dt in {0.10, 0.05, 0.02}
        5. simulation length   in {60, 120, 240}
        6. random seed (small +/-5% rate jitter) in {0, 1, 2}
    Uniform onset throughout. Returns the set of admitted orderings."""
    qs = QUANTITIES
    scales = (0.6, 1.0, 1.6)
    thetas = (0.3, 0.5, 0.7)
    onsets = (0.01, 0.02, 0.05)
    dts = (0.10, 0.05, 0.02)
    tmaxs = (60.0, 120.0, 240.0)
    seeds = (0, 1, 2)
    admitted = {}
    n_points = 0
    for dt in dts:                       # group by integration settings -> batch
        for t_max in tmaxs:
            rows_rates, rows_onset, rows_theta = [], [], []
            for g in scales:
                for theta in thetas:
                    for r0 in onsets:
                        for seed in seeds:
                            rng = np.random.default_rng(seed)
                            jit = 1.0 + 0.05 * (rng.random(len(qs)) - 0.5) * 2
                            rows_rates.append(g * jit)
                            rows_onset.append(np.full(len(qs), r0))
                            rows_theta.append(theta)
            times = milestone_times_batch(np.array(rows_rates),
                                          np.array(rows_onset),
                                          theta=np.array(rows_theta),
                                          t_max=t_max, dt=dt)
            for order in order_from_times(times):
                admitted[order] = admitted.get(order, 0) + 1
                n_points += 1
    return {"n_points": n_points, "admitted_orderings": admitted,
            "n_admitted": len(admitted), "n_permutations_total": N_PERMUTATIONS}


def nonuniform_admitted(n_samples=8000, seed=0, theta=0.5, onset_hi=0.45,
                        t_max=200.0, dt=0.05):
    """Non-uniform onset sampled BELOW threshold, rates equal. Which of the 120
    orderings appear? The theorem permits only a restricted subset."""
    rng = np.random.default_rng(seed)
    qs = QUANTITIES
    onset_arr = rng.uniform(0.0, onset_hi, size=(n_samples, len(qs)))
    rates_arr = np.ones((n_samples, len(qs)))
    times = milestone_times_batch(rates_arr, onset_arr, theta=theta,
                                  t_max=t_max, dt=dt)
    admitted = {}
    for order in order_from_times(times):
        admitted[order] = admitted.get(order, 0) + 1
    return admitted


def forbidden_reversal_holds(admitted):
    """Structural invariants guaranteed for below-threshold onset: the maximally
    gated channel is never first, the ungated root is never last, and the fully
    reversed Saito Loop never occurs."""
    reversed_loop = tuple(reversed(CLINICAL_SAITO_LOOP))
    return {
        "pi_vslow_never_first": all(o[0] != "pi_vslow" for o in admitted),
        "pi_s_never_last": all(o[-1] != "pi_s" for o in admitted),
        "fully_reversed_forbidden": reversed_loop not in admitted,
    }


# =============================================================================
# 6. COMPETING MODELS
# =============================================================================
# independent_rate  : no gating -> dr_k/dt = c_k (1 - r_k); any of the 120
#                     orderings can be produced (no prerequisite constraint).
# common_severity   : one latent severity drives every channel -> all hit
#                     threshold together (no dissociation).

def independent_rate_admitted(levels=(0.6, 1.0, 1.6), theta=0.5,
                              onset_value=0.02, t_max=80.0, dt=0.05):
    """Admitted orderings for the independent-rate model on the 3**5 rate
    grid (uniform onset)."""
    qs = QUANTITIES
    admitted = {}
    for combo in itertools.product(levels, repeat=len(qs)):
        c = np.array(combo, float)
        r = np.full(len(qs), onset_value, float)
        times = np.full(len(qs), np.inf)
        for i in range(int(round(t_max / dt))):
            r = r + dt * c * (1.0 - r)
            t = (i + 1) * dt
            for k in range(len(qs)):
                if times[k] == np.inf and r[k] >= theta:
                    times[k] = t
        order = tuple(qs[i] for i in np.argsort(times, kind="stable"))
        admitted[order] = admitted.get(order, 0) + 1
    return admitted


def common_severity_dissociation(theta=0.5, t_max=80.0, dt=0.05):
    """Milestone spread (max-min) for the common-severity model (~0)."""
    qs = QUANTITIES
    s = 0.02
    times = np.full(len(qs), np.inf)
    for i in range(int(round(t_max / dt))):
        s = s + dt * 1.0 * (1.0 - s)
        t = (i + 1) * dt
        for k in range(len(qs)):
            if times[k] == np.inf and s >= theta:
                times[k] = t
    return {"milestone_times": times, "spread": float(np.max(times) - np.min(times))}


def saito_reproduces_dissociation():
    """Saito model milestone spread under uniform onset (dissociations present)."""
    qs = QUANTITIES
    T, S = simulate({q: 1.0 for q in qs}, {q: 0.02 for q in qs},
                    t_max=80.0, dt=0.02)
    times = milestone_times(T, S, theta=0.5)
    return {"milestone_times": times, "spread": float(np.max(times) - np.min(times))}


# =============================================================================
# 7. FIGURES
# =============================================================================
plt.rcParams.update({
    "font.family": "DejaVu Serif", "font.size": 10, "axes.linewidth": 0.8,
    "savefig.dpi": 300, "figure.dpi": 150,
})
_COLORS = ["#1b4965", "#2a9d8f", "#e9c46a", "#e76f51", "#9b2226"]
_CMAP = {q: _COLORS[i] for i, q in enumerate(QUANTITIES)}


def figure1_dag(reach, order, G, path):
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.2, 3.6),
                                   gridspec_kw={"width_ratios": [1.0, 1.25]})
    labels = [LABELS[q] for q in order]
    axA.imshow(reach, cmap="Greys", vmin=0, vmax=1, aspect="equal")
    axA.set_xticks(range(len(order))); axA.set_yticks(range(len(order)))
    axA.set_xticklabels(labels); axA.set_yticklabels(labels)
    axA.set_xlabel("collapses information for"); axA.set_ylabel("zeroing")
    for i in range(len(order)):
        for j in range(len(order)):
            if reach[i, j]:
                axA.text(j, i, "1", ha="center", va="center",
                         color="white", fontsize=9)
    axA.set_title("a  Zeroing dependency (reachability)", loc="left",
                  fontsize=10, fontweight="bold")
    axB.axis("off")
    pos = {q: (i, 0) for i, q in enumerate(order)}
    for q in order:
        axB.scatter(*pos[q], s=1500, color=_CMAP[q], zorder=3,
                    edgecolors="black", linewidths=0.8)
        axB.text(pos[q][0], pos[q][1], LABELS[q], ha="center", va="center",
                 color="white", fontsize=11, zorder=4)
    for a, b in G.edges():
        axB.add_patch(FancyArrowPatch(pos[a], pos[b], arrowstyle="-|>",
                      mutation_scale=16, lw=1.6, color="#333333",
                      shrinkA=22, shrinkB=22, zorder=2))
    axB.set_xlim(-0.6, len(order) - 0.4); axB.set_ylim(-1, 1)
    axB.set_title("b  Prerequisite DAG — unique topological order",
                  loc="left", fontsize=10, fontweight="bold")
    fig.tight_layout(); _finish(fig, path)


def figure2_recovery(path):
    rates = {q: 1.0 for q in QUANTITIES}
    onset = {q: 0.02 for q in QUANTITIES}
    T, S = simulate(rates, onset, t_max=40.0, dt=0.02)
    theta = 0.5
    times = milestone_times(T, S, theta=theta)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for k, q in enumerate(QUANTITIES):
        ax.plot(T, S[:, k], color=_CMAP[q], lw=2.0, label=LABELS[q])
        if np.isfinite(times[k]):
            ax.plot([times[k]], [theta], "o", color=_CMAP[q],
                    mec="black", mew=0.7, ms=7, zorder=5)
    ax.axhline(theta, color="grey", ls="--", lw=0.8)
    ax.text(T[-1], theta + 0.02, "milestone threshold", ha="right",
            va="bottom", fontsize=8, color="grey")
    ax.set_xlabel("time (arbitrary units)")
    ax.set_ylabel("recovery state  $r_k(t)$"); ax.set_ylim(0, 1.02)
    ax.set_title("Nested availability forbids crossings → Saito Loop order",
                 fontsize=10, fontweight="bold")
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout(); _finish(fig, path)


def figure3_onset(path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.8), sharey=True)
    theta = 0.5
    rates = {q: 1.0 for q in QUANTITIES}
    T, S = simulate(rates, {q: 0.02 for q in QUANTITIES}, t_max=40.0, dt=0.02)
    for k, q in enumerate(QUANTITIES):
        ax1.plot(T, S[:, k], color=_CMAP[q], lw=1.8, label=LABELS[q])
    ax1.axhline(theta, color="grey", ls="--", lw=0.8)
    ax1.set_title("a  Uniform onset → canonical Saito Loop", loc="left",
                  fontsize=10, fontweight="bold")
    ax1.set_xlabel("time"); ax1.set_ylabel("recovery state $r_k(t)$")
    ax1.legend(loc="lower right", frameon=False, fontsize=8)
    onset_nu = {"pi_s": 0.02, "beta": 0.02, "pi_m": 0.02,
                "pi_vfast": 0.44, "pi_vslow": 0.02}
    T2, S2 = simulate(rates, onset_nu, t_max=40.0, dt=0.02)
    for k, q in enumerate(QUANTITIES):
        ax2.plot(T2, S2[:, k], color=_CMAP[q], lw=1.8, label=LABELS[q])
    ax2.axhline(theta, color="grey", ls="--", lw=0.8)
    ax2.set_title("b  Non-uniform onset → permitted reversal", loc="left",
                  fontsize=10, fontweight="bold")
    ax2.set_xlabel("time")
    fig.tight_layout(); _finish(fig, path)


def figure4_competing(grid_res, indep_admitted, common_res, saito_res, path):
    """Two-tier model-space audit figure (revised layout).

    TOP row  — the three competing accounts side by side on the two properties
               that discriminate them: uniqueness of the admitted order (a) and
               milestone dissociation (b). Independent-rate fails uniqueness;
               common-severity fails dissociation; only the Saito Loop passes
               both, so 'parameter-independent single order' is the headline.
    BOTTOM row — 729 pre-declared parameter combinations map to one and the same
               order: the number 729 is shown as ROBUSTNESS evidence, not as the
               claim itself.
    """
    import matplotlib.gridspec as _gs
    from matplotlib.colors import ListedColormap
    fig = plt.figure(figsize=(9.2, 6.5))
    gs = _gs.GridSpec(2, 2, height_ratios=[1.0, 0.85], hspace=0.55, wspace=0.33)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, :])

    # (a) uniqueness of the admitted milestone order
    names = ["Saito\nLoop", "Independent-\nrate", "Common-\nseverity"]
    vals = [grid_res["n_admitted"], len(indep_admitted), 1]
    bars = axA.bar(names, vals, color=["#2a9d8f", "#e76f51", "#6c757d"],
                   edgecolor="black", linewidth=0.7)
    tot = grid_res["n_permutations_total"]
    axA.axhline(tot, color="#adb5bd", ls="--", lw=0.9)
    axA.text(2.45, tot, f"{tot} possible", ha="right", va="bottom",
             fontsize=8, color="#6c757d")
    for b, v in zip(bars, vals):
        axA.text(b.get_x() + b.get_width() / 2, v + 1.5, str(v),
                 ha="center", va="bottom", fontsize=9)
    axA.set_ylabel("admitted milestone orderings")
    axA.set_ylim(0, tot * 1.16)
    axA.set_title("a  Uniqueness of the recovery order", loc="left",
                  fontsize=10, fontweight="bold")

    # (b) milestone dissociation (spread between first and last milestone)
    labels = ["Saito\nLoop", "Common-\nseverity"]
    spreads = [saito_res["spread"], common_res["spread"]]
    bars2 = axB.bar(labels, spreads, color=["#2a9d8f", "#6c757d"],
                    edgecolor="black", linewidth=0.7)
    for b, v in zip(bars2, spreads):
        axB.text(b.get_x() + b.get_width() / 2, v + 0.04, f"{v:.2f}",
                 ha="center", va="bottom", fontsize=9)
    axB.set_ylabel("milestone spread (max − min time)")
    axB.set_ylim(0, max(spreads) * 1.28 + 0.2)
    axB.set_title("b  Dissociation between milestones", loc="left",
                  fontsize=10, fontweight="bold")

    # (c) 729 -> same order, shown as robustness
    n_pts = grid_res.get("n_points", 729)
    side = int(round(n_pts ** 0.5))
    axC.imshow(np.ones((side, side)), cmap=ListedColormap(["#2a9d8f"]),
               vmin=0, vmax=1, aspect="auto")
    for k in range(side + 1):
        axC.axvline(k - 0.5, color="white", lw=0.4)
        axC.axhline(k - 0.5, color="white", lw=0.4)
    axC.set_xticks([]); axC.set_yticks([])
    axC.set_title("c  Robustness — 729 pre-declared parameter combinations give "
                  "one and the same order", loc="left",
                  fontsize=10, fontweight="bold")
    axC.text(side / 2 - 0.5, side / 2 - 0.5,
             r"$\pi_s \rightarrow \beta \rightarrow \pi_m \rightarrow "
             r"\pi_{v\_\mathrm{fast}} \rightarrow \pi_{v\_\mathrm{slow}}$"
             "\n\n"
             f"distinct orderings across the grid: {grid_res['n_admitted']} "
             f"of {tot}",
             ha="center", va="center", color="white", fontsize=12,
             fontweight="bold")
    axC.set_xlabel("each cell = one parameter combination "
                   "(6 nuisance hyperparameters × 3 levels = 729)")
    _finish(fig, path)


# =============================================================================
# 8. DRIVER — run everything and print a verification report
# =============================================================================
def main(fig_dir="."):
    report = {}
    print("=" * 70)
    print("SAITO LOOP — deductive reproduction (single file)")
    print("=" * 70)

    # Stage 1
    s1 = derived_order()
    reach, G, order, is_unique = (s1["reachability"], s1["graph"],
                                  s1["order"], s1["is_unique"])
    matches = order == CLINICAL_SAITO_LOOP
    print("\n[Stage 1] Identifiability -> DAG -> topological order")
    print("  recovered edges :", [f"{a}->{b}" for a, b in G.edges()])
    print("  topological order:", " -> ".join(order))
    print("  unique ordering :", is_unique)
    print("  matches clinical Saito Loop (held out):", matches)
    report["stage1"] = {"order": list(order), "is_unique": is_unique,
                        "matches_clinical": matches}

    # Stage 2
    T, S = simulate({q: 1.0 for q in QUANTITIES},
                    {q: 0.02 for q in QUANTITIES}, t_max=80.0, dt=0.02)
    order2, times = milestone_order(T, S, theta=0.5)
    nocross = no_crossings(S)
    print("\n[Stage 2] Recovery dynamics under uniform onset")
    print("  milestone order :", " -> ".join(order2))
    print("  no crossings    :", nocross)
    report["stage2"] = {"order": list(order2), "no_crossings": nocross,
                        "milestone_times": [round(float(x), 3) for x in times]}

    # Robustness
    grid = robustness_grid()
    print("\n[Robustness] pre-declared 729-point grid")
    print(f"  points evaluated   : {grid['n_points']}")
    print(f"  distinct orderings : {grid['n_admitted']} of "
          f"{grid['n_permutations_total']}")
    only = list(grid["admitted_orderings"].keys())
    if grid["n_admitted"] == 1:
        print("  admitted ordering  :", " -> ".join(only[0]))
    report["robustness"] = {"n_points": grid["n_points"],
                            "n_admitted": grid["n_admitted"],
                            "n_total": grid["n_permutations_total"]}

    # Competing models
    indep = independent_rate_admitted()
    common = common_severity_dissociation()
    saito_d = saito_reproduces_dissociation()
    print("\n[Competing models]")
    print(f"  independent-rate: {len(indep)} distinct orderings")
    print(f"  common-severity : spread = {common['spread']:.3f} (no dissociation)")
    print(f"  Saito Loop      : spread = {saito_d['spread']:.3f} (dissociations)")
    report["competing"] = {"independent_rate_n_orderings": len(indep),
                           "common_severity_spread": round(common["spread"], 3),
                           "saito_spread": round(saito_d["spread"], 3)}

    # Non-uniform onset
    nu = nonuniform_admitted()
    forb = forbidden_reversal_holds(nu)
    print("\n[Non-uniform onset] below-threshold sweep")
    print(f"  admitted orderings      : {len(nu)} of {N_PERMUTATIONS}")
    print(f"  pi_vslow never first    : {forb['pi_vslow_never_first']}")
    print(f"  pi_s never last         : {forb['pi_s_never_last']}")
    print(f"  fully-reversed forbidden: {forb['fully_reversed_forbidden']}")
    report["nonuniform"] = {"n_admitted": len(nu), "n_total": N_PERMUTATIONS,
                            **forb}

    # Figures
    os.makedirs(fig_dir, exist_ok=True)
    print("\n[Figures] writing to", fig_dir)
    figure1_dag(reach, order, G, f"{fig_dir}/Figure1_DAG.png")
    figure2_recovery(f"{fig_dir}/Figure2_Recovery.png")
    figure3_onset(f"{fig_dir}/Figure3_Onset.png")
    figure4_competing(grid, indep, common, saito_d,
                      f"{fig_dir}/Figure4_Competing.png")
    print("  wrote Figure1_DAG.png .. Figure4_Competing.png")

    # Verdict
    ok = (is_unique and matches and nocross
          and order2 == CLINICAL_SAITO_LOOP
          and grid["n_admitted"] == 1
          and all(forb.values()))
    print("\n" + "=" * 70)
    print("OVERALL: all deductive claims reproduced =", ok)
    print("=" * 70)
    report["overall_pass"] = bool(ok)
    with open(f"{fig_dir}/verification_report.json", "w") as fh:
        json.dump(report, fh, indent=2)
    return report

# ============================ PART B ============================
from dataclasses import dataclass
from itertools import product, permutations
import csv, json
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

NODES: Tuple[str, ...] = ("pi_s", "beta", "pi_m", "pi_v_fast", "pi_v_slow")
ANCESTORS: Tuple[Tuple[int, ...], ...] = ((), (0,), (1,), (2,), (3,))

@dataclass(frozen=True)
class Family:
    name: str; direct_prerequisite: bool; shared_latent: bool

FAMILIES: Tuple[Family, ...] = (
    Family("independent_rate", False, False),
    Family("common_severity", False, True),
    Family("saito_prerequisite", True, False),
    Family("hybrid", True, True),
)

@dataclass(frozen=True)
class Params:
    rate_scale: float; threshold: float; onset: float
    dt: float; severity_floor: float; severity_power: float

@dataclass
class AuditResult:
    family: str; grid_points: int = 0; canonical_points: int = 0
    dissociated_points: int = 0; directional_points: int = 0
    all_signature_points: int = 0; distinct_orders: set = None
    def __post_init__(self):
        if self.distinct_orders is None: self.distinct_orders = set()

def latent_severity(r, floor, power):
    return floor + (1.0 - floor) * ((sum(r)/len(r)) ** power)

def _b_availability(i, r, family, p):
    a = 1.0
    if family.direct_prerequisite:
        for j in ANCESTORS[i]: a *= max(0.0, min(1.0, r[j]))
    if family.shared_latent:
        a *= latent_severity(r, p.severity_floor, p.severity_power)
    return a

def rates_for_family(family, scale, independent_perm=None):
    if family.name == "independent_rate":
        if independent_perm is None:
            independent_perm = (0.70, 0.85, 1.00, 1.15, 1.30)
        return tuple(scale * x for x in independent_perm)
    return (scale,) * len(NODES)

def derivative(r, family, p, rates):
    return [rates[i]*_b_availability(i, r, family, p)*(1.0-ri) for i, ri in enumerate(r)]

def rk4_step(r, h, family, p, rates):
    k1 = derivative(r, family, p, rates)
    r2 = [min(1.0,max(0.0,x+0.5*h*dx)) for x,dx in zip(r,k1)]
    k2 = derivative(r2, family, p, rates)
    r3 = [min(1.0,max(0.0,x+0.5*h*dx)) for x,dx in zip(r,k2)]
    k3 = derivative(r3, family, p, rates)
    r4 = [min(1.0,max(0.0,x+h*dx)) for x,dx in zip(r,k3)]
    k4 = derivative(r4, family, p, rates)
    return [min(1.0,max(0.0,x+h*(a+2*b+2*c+d)/6.0)) for x,a,b,c,d in zip(r,k1,k2,k3,k4)]

def _b_simulate(family, p, initial, independent_perm=None, t_max=200.0):
    rates = rates_for_family(family, p.rate_scale, independent_perm)
    r = list(initial); milestones = [None]*len(NODES); t = 0.0
    while t <= t_max:
        for i, ri in enumerate(r):
            if milestones[i] is None and ri >= p.threshold: milestones[i] = t
        if all(x is not None for x in milestones): break
        r = rk4_step(r, p.dt, family, p, rates); t += p.dt
    if any(x is None for x in milestones):
        raise RuntimeError(f"Milestone not reached for {family.name}")
    ms = [float(x) for x in milestones]
    order = tuple(sorted(range(len(NODES)), key=lambda i: (ms[i], i)))
    return ms, order

def is_canonical(ms, tol=1e-8):
    return all(ms[i] <= ms[i+1]+tol for i in range(len(ms)-1)) and (ms[-1]-ms[0] > tol)

def is_dissociated(ms, tol=0.05):
    return max(ms)-min(ms) > tol

def directional_signature(family, p, independent_perm=None, spare_level=0.25,
                          zero_tol=5e-3, effect_tol=5e-3):
    base_initial = [p.onset]*len(NODES)
    base_ms, _ = _b_simulate(family, p, base_initial, independent_perm)
    for j in range(len(NODES)):
        init = base_initial.copy(); init[j] = max(spare_level, p.onset)
        pert_ms, _ = _b_simulate(family, p, init, independent_perm)
        advance = [b-q for b,q in zip(base_ms, pert_ms)]
        if any(abs(advance[i]) > zero_tol for i in range(j)): return False
        if any(advance[i] <= effect_tol for i in range(j, len(NODES))): return False
    return True

def parameter_grid() -> Iterable[Params]:
    for vals in product((0.10,0.20,0.40),(0.40,0.50,0.60),(0.01,0.03,0.05),
                        (0.10,0.20,0.40),(0.10,0.25,0.40),(0.50,1.00,2.00)):
        yield Params(*vals)

def audit_family(family):
    result = AuditResult(family=family.name); grid = list(parameter_grid())
    independent_perms = list(permutations((0.70,0.85,1.00,1.15,1.30)))
    for idx, p in enumerate(grid):
        perm = independent_perms[idx % len(independent_perms)] if family.name=="independent_rate" else None
        ms, order = _b_simulate(family, p, [p.onset]*len(NODES), perm)
        result.grid_points += 1; result.distinct_orders.add(order)
        c = is_canonical(ms); d = is_dissociated(ms); s = directional_signature(family, p, perm)
        result.canonical_points += int(c); result.dissociated_points += int(d)
        result.directional_points += int(s); result.all_signature_points += int(c and d and s)
    return result

def check_partition():
    observed = {(f.direct_prerequisite, f.shared_latent) for f in FAMILIES}
    expected = {(False,False),(False,True),(True,False),(True,True)}
    assert observed == expected and len(FAMILIES) == 4, "within-C partition broken"

def _audit_B_main():
    check_partition()
    results = [audit_family(f) for f in FAMILIES]
    print("family,grid,canonical,dissociation,directional,all3,distinct_orders")
    for r in results:
        print(f"{r.family},{r.grid_points},{r.canonical_points},{r.dissociated_points},"
              f"{r.directional_points},{r.all_signature_points},{len(r.distinct_orders)}")
    return results

# ============================ PART C ============================
from dataclasses import dataclass
from enum import Enum
from itertools import product


class StructuralEdit(str, Enum):
    ISOMORPHISM = "isomorphism_or_relabel"
    COARSENING = "node_delete_or_contract"
    REFINEMENT = "node_insert_or_split"
    REWIRING = "edge_add_delete_or_reverse"


@dataclass(frozen=True)
class OneEditCandidate:
    node_removed_or_contracted: bool = False
    node_inserted_or_split: bool = False
    edge_edited: bool = False


def classify_one_edit(c: OneEditCandidate) -> StructuralEdit:
    flags = (c.node_removed_or_contracted, c.node_inserted_or_split, c.edge_edited)
    if sum(flags) > 1:
        raise ValueError("Not in the one-edit neighbourhood: multiple edits active.")
    if c.node_removed_or_contracted: return StructuralEdit.COARSENING
    if c.node_inserted_or_split:     return StructuralEdit.REFINEMENT
    if c.edge_edited:                return StructuralEdit.REWIRING
    return StructuralEdit.ISOMORPHISM


@dataclass(frozen=True)
class CouplingFamily:
    prerequisite_gating: bool
    shared_severity: bool
    @property
    def name(self) -> str:
        return {
            (False, False): "independent_rate",
            (False, True):  "common_severity",
            (True, False):  "prerequisite_limited_saito",
            (True, True):   "hybrid",
        }[(self.prerequisite_gating, self.shared_severity)]


def verify_partitions() -> dict:
    structural = tuple(classify_one_edit(c) for c in (
        OneEditCandidate(),
        OneEditCandidate(node_removed_or_contracted=True),
        OneEditCandidate(node_inserted_or_split=True),
        OneEditCandidate(edge_edited=True),
    ))
    assert len(set(structural)) == 4 and set(structural) == set(StructuralEdit)

    families = tuple(CouplingFamily(p, s) for p, s in product((False, True), repeat=2))
    pairs = {(f.prerequisite_gating, f.shared_severity) for f in families}
    assert len(families) == 4 and pairs == {(False, False), (False, True),
                                            (True, False), (True, True)}
    return {
        "structural_partition": [x.value for x in structural],
        "coupling_partition": [f.name for f in families],
        "scope": {
            "structural": "architectures at graph-edit distance <=1 from M",
            "dynamical": "five-state monotone recovery models using only P and S",
            "universal_uniqueness_claimed": False,
        },
    }


def _classify_C_main() -> dict:
    r = verify_partitions()
    print("FORMAL CLASSIFICATION AUDIT: PASS")
    print("Structural one-edit classes:", ", ".join(r["structural_partition"]))
    print("Dynamical coupling families:", ", ".join(r["coupling_partition"]))
    print("universal_uniqueness_claimed:", r["scope"]["universal_uniqueness_claimed"])
    return r


# =============================================================================
# UNIFIED DRIVER - runs Parts A, B, C and prints one verification report
# =============================================================================
def run_all(fig_dir="."):
    print("#" * 72)
    print("# PART A - two-stage deduction + robustness + figures")
    print("#" * 72)
    rep = main(fig_dir)                      # Part A main(), inlined above

    print("\n" + "#" * 72)
    print("# PART B - reference dynamical audit of the four (P,S) families")
    print("#" * 72)
    results = _audit_B_main()
    passers = [r.family for r in results if r.all_signature_points == r.grid_points]
    print("\nSole family satisfying all three signatures:",
          ", ".join(passers) if passers else "none")

    print("\n" + "#" * 72)
    print("# PART C - formal bounded classification (Theorems 1 and 2)")
    print("#" * 72)
    rc = _classify_C_main()

    ok_A = rep.get("overall_pass", False)
    ok_B = passers == ["saito_prerequisite"]
    ok_C = (len(set(rc["structural_partition"])) == 4
            and len(set(rc["coupling_partition"])) == 4
            and rc["scope"]["universal_uniqueness_claimed"] is False)
    print("\n" + "=" * 72)
    print(f"OVERALL: Part A deductions = {ok_A}; "
          f"Part B sole passer = Saito = {ok_B}; "
          f"Part C bounded partitions verified = {ok_C}")
    print("Interpretation: CONDITIONAL uniqueness within N1(M) and C. "
          "NOT universal uniqueness.")
    print("=" * 72)
    return ok_A and ok_B and ok_C


def show_saved_figures(fig_dir="."):
    """Manually display the saved PNGs inline. Handy if the analysis was run
    with `!python ...` (a subprocess that cannot draw into the notebook):
    run the script that way, then call show_saved_figures() in a new cell."""
    try:
        from IPython.display import Image, display
    except Exception:
        print("Not in a notebook; PNGs are in", os.path.abspath(fig_dir))
        return
    for name in ("Figure1_DAG.png", "Figure2_Recovery.png",
                 "Figure3_Onset.png", "Figure4_Competing.png"):
        p = os.path.join(fig_dir, name)
        if os.path.exists(p):
            display(Image(filename=p))


if __name__ == "__main__":
    run_all(_resolve_out_dir())
