#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
recovery_order_verification.py
================================================================================
Recovery order in adaptive systems is set by dependency structure.
Numerical validation of the structure theorem (Methods, Theorems 1-3) and of its
Fisher realization (Theorems A and B), together with the three instantiations:
a chain (hierarchical recovery in catatonia -> the Saito Loop), a branch
(post-injury motor recovery -> a recovery poset) and a NON-PROBABILISTIC branch
(curriculum learning in an artificial agent -> a recovery poset).

The theory is stated at the level of a dependency graph: a recovery-order system
obeys four axioms -- (A1) capacities with recovered fractions, (A2) a common
impairment, (A3) an acyclic prerequisite dependency graph, (A4) a monotone,
deficit-closing learning rule.  The structure theorem states that (i) onset
precedence is UNCONDITIONAL and (ii) in the gating-dominated regime (rate
constants non-increasing along edges, c_k <= c_j) the admissible milestone
orders are EXACTLY the linear extensions of the graph.  The Fisher information
of a probabilistic model is ONE realization of these axioms (its support gives
the graph, its magnitude gives the learning drive); the curriculum-learning case
realizes the same axioms with no likelihood at all.

This single, self-contained file *validates* (it does not prove) the analytical
results of the manuscript.  The proofs live in the paper (Methods, Theorems A, B
and 1-3); the numbers below confirm that concrete minimal models behave exactly
as those theorems require.

FIGURE MAP (revised manuscript: three main display items, four supplementary)
    Fig. 1  Fisher information factorizes through the dependency graph   (Thm A)
    Fig. 2  the recovery law is derived, not assumed                     (Thm B)
    Fig. 3  admissible orders are the linear extensions -- and where the law ends
              (a) chain: one order across a pre-declared grid
              (b) branch: a recovery poset (both linear extensions)
              (c) boundary of prerequisite-gating dominance
    S. 1  the derived saturating factor (1 - r_k)
    S. 2  the counterexample: a fast descendant overtakes its prerequisite
    S. 3  post-injury motor recovery                            (instantiation 2)
    S. 4  curriculum learning: a NON-Fisher branch realization  (instantiation 3)

Note on the identifiability graph.  The ancestor sets anc(k) that define the gate
ARE the analytic prerequisite structure of Theorem A; they are supplied as the
premise, not searched for.  The checks therefore confirm that the Fisher
information has the DERIVED MAGNITUDE and the DERIVED SUPPORT (its zero-pattern
matches the graph); they do not "discover" the edges empirically.  The channel
individuation itself is likewise assumed, and named as such in the manuscript
(it is the theory's central open problem).

What is checked
---------------
THEOREM A  (multiplicative factorization of Fisher information)
        I_k(r) = c_k * prod_{j in anc(k)} r_j
    confirmed three independent ways: (A1) closed form of the gated Gaussian
    channel; (A2) Monte-Carlo estimate as Var(score); (A3) the product itself.
    The ZERO-PATTERN (support) of I_k is confirmed to reproduce the prerequisite
    graph pi_s -> beta -> pi_m -> pi_v_fast -> pi_v_slow.

THEOREM B  (from Fisher information to the multiplicative recovery law)
    (B1)  posterior precision accumulates Fisher information:  dP_k/dt = I_k(t).
    (B2)  r_k = 1 - exp(-(P_k - P0)/s) is the UNIQUE link for which the marginal
          recovery rate is proportional to the deficit -> the factor (1 - r_k).
    (B3)  integrating dP_k/dt = c_k * prod r_j and mapping through that link
          reproduces  dr_k/dt = c_k' * A_k * (1 - r_k),  A_k = prod r_j.

THEOREMS 1-3  (the ordering law)
    (a) ONSET precedence is UNCONDITIONAL: with the gate vanishing at baseline
        (r0 = 0) no capacity leaves baseline before its prerequisites, for any
        rate constants.
    (b) MILESTONE precedence holds in the GATING-DOMINATED REGIME, whose clean
        sufficient condition is c_k <= c_j along every edge.  Section 3c
        reproduces the counterexample that breaks the unconditional claim
        (r0 = 0.5, c_parent = 1, c_child = 100, which satisfies A1-A4 and
        Proposition A.3 yet reverses the edge), confirms the sufficient
        condition on a 5-node chain, confirms onset precedence is unconditional,
        and maps the reversal boundary (Fig. 3c).

Dependencies: numpy, matplotlib.  Networkx is NOT required.

Run:
    python recovery_order_verification.py            # report + figures here
    python recovery_order_verification.py ./out      # figures to ./out
================================================================================
"""
from __future__ import annotations

import sys
import os
import json
import itertools

import numpy as np

import matplotlib


def _in_notebook():
    """True inside Colab / Jupyter, where figures should render inline."""
    if "google.colab" in sys.modules:
        return True
    try:
        from IPython import get_ipython
        ip = get_ipython()
        return ip is not None and ip.__class__.__name__ == "ZMQInteractiveShell"
    except Exception:
        return False


IN_NOTEBOOK = _in_notebook()
if not IN_NOTEBOOK:
    # Headless / plain-script use: write PNGs without needing a display.
    matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _finalize(fig, path, dpi=None):
    """Save every figure to disk AND, in a notebook, show it inline.

    Replaces the old ``savefig(); plt.close()`` idiom that silently suppressed
    all figures in Colab.  ``path``'s directory is created if missing.
    """
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if dpi is not None:
        fig.savefig(path, bbox_inches="tight", dpi=dpi)
    else:
        fig.savefig(path, bbox_inches="tight")
    if IN_NOTEBOOK:
        plt.show()            # render inline in Colab/Jupyter
    plt.close(fig)


# =============================================================================
# 0. MODEL: five precisions and the prerequisite (identifiability) graph
# =============================================================================
# The five computational precisions and the ONLY structural commitment of the
# theory: a chain of prerequisite (direct identifiability-dependency) edges.
# No recovery order, milestone ranking or temporal constraint is entered here.

QUANTITIES = ["pi_s", "beta", "pi_m", "pi_vfast", "pi_vslow"]

LABELS = {
    "pi_s":     r"$\pi_s$",
    "beta":     r"$\beta$",
    "pi_m":     r"$\pi_m$",
    "pi_vfast": r"$\pi_{v\_\mathrm{fast}}$",
    "pi_vslow": r"$\pi_{v\_\mathrm{slow}}$",
}

PREREQUISITE_EDGES = [
    ("pi_s", "beta"),
    ("beta", "pi_m"),
    ("pi_m", "pi_vfast"),
    ("pi_vfast", "pi_vslow"),
]

# Held out for validation only; never used to derive anything below.
CLINICAL_SAITO_LOOP = ("pi_s", "beta", "pi_m", "pi_vfast", "pi_vslow")

N = len(QUANTITIES)


def index(name):
    return QUANTITIES.index(name)


def ancestors():
    """Transitive identifiability ancestors implied by the prerequisite edges."""
    parents = {q: set() for q in QUANTITIES}
    for a, b in PREREQUISITE_EDGES:
        parents[b].add(a)
    anc = {q: set() for q in QUANTITIES}
    changed = True
    while changed:                              # fixed-point transitive closure
        changed = False
        for q in QUANTITIES:
            new = set(parents[q])
            for p in list(parents[q]):
                new |= anc[p]
            if new != anc[q]:
                anc[q] = new
                changed = True
    return anc


ANC = ancestors()
# Sort the ancestor indices.  ANC[q] is a set of *strings*, and Python randomizes
# string hashing per process, so iterating the set directly makes the order (and
# therefore the order in which fisher_montecarlo consumes random draws) depend on
# PYTHONHASHSEED.  Sorting pins it, making every number below bit-reproducible.
ANC_IDX = [sorted(index(p) for p in ANC[q]) for q in QUANTITIES]


# =============================================================================
# 1. THEOREM A: the gated Gaussian channel and Fisher-information factorization
# =============================================================================
# Minimal generative model for the channel about a single precision theta_k.
#
#   * Each ancestor j is "online" independently with probability r_j
#     (Axiom I: permissibility independence).  The gate is
#         g_k = prod_{j in anc(k)} Bernoulli(r_j),   P(g_k = 1) = prod r_j.
#   * If g_k = 1 the observation is informative about theta_k:
#         y ~ Normal(0, 1/theta_k)                 (theta_k is a precision)
#     whose per-observation Fisher information about theta_k is
#         c_k(theta_k) = 1/(2 theta_k^2).
#   * If g_k = 0 the observation is drawn from a fixed reference channel that
#     does NOT depend on theta_k, contributing zero Fisher information.
#
# Because the gate's marginal does not depend on theta_k it is ANCILLARY, so the
# (observed-gate) Fisher information is the gate-averaged conditional
# information:  I_k = E_g[ I(theta_k | g) ] = c_k * P(g = 1) = c_k * prod r_j.


def per_informative_fisher(theta_k):
    """Closed-form Fisher information of one *informative* Normal(0, 1/theta_k)
    observation about the precision theta_k:  I = 1/(2 theta_k^2)."""
    return 1.0 / (2.0 * theta_k ** 2)


def fisher_analytic(levels, thetas):
    """(A1)+(A3) Closed-form gated Fisher information for every precision:
        I_k = c_k(theta_k) * prod_{j in anc(k)} r_j.
    levels, thetas: dict name -> value in [0,1] and > 0 respectively."""
    info = {}
    for k, q in enumerate(QUANTITIES):
        gate = 1.0
        for p in ANC_IDX[k]:
            gate *= levels[QUANTITIES[p]]
        info[q] = per_informative_fisher(thetas[q]) * gate
    return info


def fisher_montecarlo(q, levels, thetas, n_samples=1_200_000, seed=0):
    """(A2) Monte-Carlo Fisher information about theta_q as the variance of the
    score of the *observed-gate* gated channel.  Confirms fisher_analytic."""
    rng = np.random.default_rng(seed)
    theta = thetas[q]
    k = index(q)
    # Draw the independent enabling Bernoullis for the ancestors of q.
    if ANC_IDX[k]:
        gate = np.ones(n_samples, dtype=float)
        for p in ANC_IDX[k]:
            gate *= (rng.random(n_samples) < levels[QUANTITIES[p]]).astype(float)
    else:
        gate = np.ones(n_samples, dtype=float)          # root: always informative
    # Informative branch:  y ~ Normal(0, 1/theta).  (Reference branch score = 0.)
    y = rng.normal(0.0, np.sqrt(1.0 / theta), size=n_samples)
    # Score of the informative Gaussian precision:  d/dtheta log f = 1/(2 theta) - y^2/2.
    score = gate * (1.0 / (2.0 * theta) - 0.5 * y ** 2)
    return float(np.mean(score ** 2))                    # E[score^2] = Fisher info


def support_matrix(thetas=None, eps=1e-12):
    """Zero-pattern of the Fisher field: R[i, k] = 1 iff setting r_i = 0 makes
    I_k vanish while it was positive at full availability.  This reconstructs the
    prerequisite reachability from the *support* of the Fisher information."""
    if thetas is None:
        thetas = {q: 1.0 for q in QUANTITIES}
    full = {q: 1.0 for q in QUANTITIES}
    base = fisher_analytic(full, thetas)
    R = np.zeros((N, N), dtype=int)
    for i, qi in enumerate(QUANTITIES):
        lv = dict(full)
        lv[qi] = 0.0
        info = fisher_analytic(lv, thetas)
        for k, qk in enumerate(QUANTITIES):
            if qk == qi:
                continue
            if base[qk] > eps and info[qk] <= eps:
                R[i, k] = 1
    return R


def reachability_from_edges():
    """Ground-truth transitive reachability from the prerequisite edges."""
    R = np.zeros((N, N), dtype=int)
    for k, q in enumerate(QUANTITIES):
        for p in ANC_IDX[k]:
            R[p, k] = 1
    return R


def verify_theorem_A(seed=0):
    """Run all three Fisher computations over a random grid of availability
    levels and confirm they agree, and that the support = prerequisite graph."""
    rng = np.random.default_rng(seed)
    thetas = {q: 1.0 for q in QUANTITIES}          # operating precisions
    max_rel_err_analytic_vs_product = 0.0          # A1 vs A3 are identical by construction
    max_rel_err_mc = 0.0
    checked = 0
    pts = []
    for _ in range(16):
        levels = {q: float(rng.uniform(0.30, 1.0)) for q in QUANTITIES}
        Ia = fisher_analytic(levels, thetas)
        for q in QUANTITIES:
            product = per_informative_fisher(thetas[q]) * np.prod(
                [levels[QUANTITIES[p]] for p in ANC_IDX[index(q)]] or [1.0])
            # A1/A3 identity (exact):
            max_rel_err_analytic_vs_product = max(
                max_rel_err_analytic_vs_product,
                abs(Ia[q] - product) / max(product, 1e-12))
            # A2 Monte-Carlo (statistical):
            Imc = fisher_montecarlo(q, levels, thetas, seed=int(rng.integers(1 << 30)))
            rel = abs(Imc - Ia[q]) / max(Ia[q], 1e-6)
            max_rel_err_mc = max(max_rel_err_mc, rel)
            pts.append((Ia[q], Imc))
            checked += 1
    R_support = support_matrix(thetas)
    R_truth = reachability_from_edges()
    support_ok = bool(np.array_equal(R_support, R_truth))
    return {
        "checked": checked,
        "max_rel_err_analytic_vs_product": max_rel_err_analytic_vs_product,
        "max_rel_err_montecarlo": max_rel_err_mc,
        "support_reproduces_prerequisite_graph": support_ok,
        "scatter_points": pts,
        "support_matrix": R_support,
    }


# =============================================================================
# 2. THEOREM B (B1): posterior precision accumulates Fisher information
# =============================================================================
# For a Gaussian (Laplace) posterior, conjugate updating ADDS precisions:
# after an informative observation carrying Fisher information delta about
# theta_k, P_k <- P_k + delta.  Accumulating the expected flux gives
#     dP_k/dt = I_k(t).
# We verify this by comparing (i) exact conjugate precision addition over a
# stream of informative observations against (ii) the ODE dP/dt = I integrated
# with the same expected flux.


def verify_precision_accumulation(theta=1.0, gate_prob=0.6, n_obs=20000, seed=1):
    """Exact conjugate precision addition vs. dP/dt = I accumulation.
    Each candidate observation is informative w.p. gate_prob (= prod r_j) and
    then carries Fisher information c = 1/(2 theta^2)."""
    rng = np.random.default_rng(seed)
    c = per_informative_fisher(theta)              # per-informative-obs information
    P0 = 1.0
    # (i) exact: draw informative indicators, add c for each.
    informative = rng.random(n_obs) < gate_prob
    P_exact = P0 + c * np.cumsum(informative)
    # (ii) continuum: expected flux I = c * gate_prob per observation "tick".
    I = c * gate_prob
    P_ode = P0 + I * np.arange(1, n_obs + 1)
    # Relative discrepancy at the end (LLN: exact -> ode).
    rel_end = abs(P_exact[-1] - P_ode[-1]) / P_ode[-1]
    return {
        "c_per_informative_obs": c,
        "flux_I": I,
        "P_exact_final": float(P_exact[-1]),
        "P_ode_final": float(P_ode[-1]),
        "rel_discrepancy_final": float(rel_end),
        "matches": bool(rel_end < 0.02),
    }


# =============================================================================
# 3. THEOREM B (B2/B3): the saturating link and the recovery law
# =============================================================================
# Recovered fraction as a function of accumulated posterior precision Q = P - P0:
#     r = link(Q) = 1 - exp(-Q / s).
# This is the UNIQUE link with dr/dQ = (1 - r)/s (deficit-proportional gain),
# so it supplies the derived factor (1 - r).  We integrate two systems and show
# they coincide:
#     Fisher-first :  dQ_k/dt = I_k = c_k * prod_{j in anc(k)} r_j,  r_k = link(Q_k)
#     companion    :  dr_k/dt = c_k' * A_k * (1 - r_k),  A_k = prod r_j, c_k' = c_k/s

S_SCALE = 1.0


def link(Q, s=S_SCALE):
    return 1.0 - np.exp(-Q / s)


def link_inv(r, s=S_SCALE):
    return -s * np.log(np.clip(1.0 - r, 1e-15, 1.0))


def _avail(r):
    """A_k = prod_{j in anc(k)} r_j  (unity for the root)."""
    a = np.ones(N)
    for k in range(N):
        for p in ANC_IDX[k]:
            a[k] *= r[p]
    return a


def simulate_fisher_first(c, r0, s=S_SCALE, t_max=60.0, dt=0.02):
    """Integrate dQ_k/dt = c_k * prod r_j with r_k = link(Q_k) via RK4."""
    c = np.asarray(c, float)
    Q = link_inv(np.asarray(r0, float), s)          # Q(0) chosen so r(0) = r0
    n_steps = int(round(t_max / dt))
    T = np.zeros(n_steps + 1)
    Rr = np.zeros((n_steps + 1, N))
    Rr[0] = link(Q, s)

    def dQ(Qv):
        r = link(Qv, s)
        return c * _avail(r)

    for i in range(n_steps):
        k1 = dQ(Q)
        k2 = dQ(Q + 0.5 * dt * k1)
        k3 = dQ(Q + 0.5 * dt * k2)
        k4 = dQ(Q + dt * k3)
        Q = Q + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        Rr[i + 1] = link(Q, s)
        T[i + 1] = (i + 1) * dt
    return T, Rr


def simulate_companion(cprime, r0, t_max=60.0, dt=0.02):
    """Integrate the companion law dr_k/dt = c'_k * A_k * (1 - r_k) via RK4."""
    cp = np.asarray(cprime, float)
    r = np.asarray(r0, float).copy()
    n_steps = int(round(t_max / dt))
    T = np.zeros(n_steps + 1)
    Rr = np.zeros((n_steps + 1, N))
    Rr[0] = r

    def dr(rv):
        return cp * _avail(rv) * (1.0 - rv)

    for i in range(n_steps):
        k1 = dr(r)
        k2 = dr(np.clip(r + 0.5 * dt * k1, 0, 1))
        k3 = dr(np.clip(r + 0.5 * dt * k2, 0, 1))
        k4 = dr(np.clip(r + dt * k3, 0, 1))
        r = np.clip(r + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4), 0, 1)
        Rr[i + 1] = r
        T[i + 1] = (i + 1) * dt
    return T, Rr


def milestone_times(T, S, theta=0.5):
    times = np.full(N, np.inf)
    for k in range(N):
        idx = np.argmax(S[:, k] >= theta)
        if S[idx, k] >= theta:
            times[k] = T[idx]
    return times


def milestone_order(T, S, theta=0.5):
    times = milestone_times(T, S, theta)
    return tuple(QUANTITIES[i] for i in np.argsort(times, kind="stable")), times


# =============================================================================
# 3b. GENERAL DAG, Theorems 1-3: admissible orders are the linear extensions
# =============================================================================
# The ordering theorems hold for ANY finite capacity set with ANY acyclic
# identifiability graph.  We demonstrate the non-chain case on a branching DAG
#     root -> A ,  root -> B      (A and B incomparable),
# for which the two linear extensions are (root, A, B) and (root, B, A) and any
# order with the root NOT first is forbidden.  Sweeping the rate constants must
# (i) always place the root first and (ii) realize BOTH extensions.

def _ancestors_from_edges(nodes, edges):
    parents = {q: set() for q in nodes}
    for a, b in edges:
        parents[b].add(a)
    anc = {q: set() for q in nodes}
    changed = True
    while changed:
        changed = False
        for q in nodes:
            new = set(parents[q]) | set().union(*(anc[p] for p in parents[q])) \
                  if parents[q] else set()
            if new != anc[q]:
                anc[q] = new
                changed = True
    return anc


def linear_extensions(nodes, edges):
    """All topological orders (linear extensions) of the DAG on `nodes`."""
    anc = _ancestors_from_edges(nodes, edges)
    results = []

    def rec(placed, remaining):
        if not remaining:
            results.append(tuple(placed))
            return
        for x in remaining:
            if anc[x] <= set(placed):                # all ancestors already placed
                rec(placed + [x], [y for y in remaining if y != x])
    rec([], list(nodes))
    return results


def simulate_dag(nodes, edges, cprime, r0=0.02, t_max=80.0, dt=0.02):
    """Fisher-first recovery on an arbitrary DAG: dr_k/dt = c'_k * A_k * (1-r_k),
    A_k = prod_{j in anc(k)} r_j.  Returns milestone order at r = 0.5."""
    anc = _ancestors_from_edges(nodes, edges)
    idx = {q: i for i, q in enumerate(nodes)}
    anc_idx = [sorted(idx[p] for p in anc[q]) for q in nodes]   # deterministic order
    m = len(nodes)
    cp = np.asarray(cprime, float)
    r = np.full(m, float(r0))

    def avail(rv):
        a = np.ones(m)
        for k in range(m):
            for p in anc_idx[k]:
                a[k] *= rv[p]
        return a

    def dr(rv):
        return cp * avail(rv) * (1.0 - rv)

    n_steps = int(round(t_max / dt))
    Rr = np.zeros((n_steps + 1, m)); Rr[0] = r
    Tt = np.zeros(n_steps + 1)
    for i in range(n_steps):
        k1 = dr(r)
        k2 = dr(np.clip(r + 0.5 * dt * k1, 0, 1))
        k3 = dr(np.clip(r + 0.5 * dt * k2, 0, 1))
        k4 = dr(np.clip(r + dt * k3, 0, 1))
        r = np.clip(r + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4), 0, 1)
        Rr[i + 1] = r; Tt[i + 1] = (i + 1) * dt
    times = np.full(m, np.inf)
    for k in range(m):
        j = np.argmax(Rr[:, k] >= 0.5)
        if Rr[j, k] >= 0.5:
            times[k] = Tt[j]
    order = tuple(nodes[i] for i in np.argsort(times, kind="stable"))
    return order, Tt, Rr


def verify_partial_order_dag():
    nodes = ["root", "A", "B"]
    edges = [("root", "A"), ("root", "B")]
    exts = set(linear_extensions(nodes, edges))
    realized = set()
    root_always_first = True
    # sweep relative rates of the two incomparable capacities
    for cA in (0.6, 1.0, 1.6):
        for cB in (0.6, 1.0, 1.6):
            order, _, _ = simulate_dag(nodes, edges, [1.0, cA, cB])
            realized.add(order)
            if order[0] != "root":
                root_always_first = False
    all_realized_are_extensions = realized <= exts
    both_extensions_seen = exts <= realized
    # representative trajectories for the figure (the two extensions)
    _, Tfast, Rfast = simulate_dag(nodes, edges, [1.0, 1.6, 0.6])   # A before B
    _, Tslow, Rslow = simulate_dag(nodes, edges, [1.0, 0.6, 1.6])   # B before A
    return {
        "nodes": nodes, "edges": edges,
        "linear_extensions": sorted(exts),
        "realized_orders": sorted(realized),
        "all_realized_are_linear_extensions": bool(all_realized_are_extensions),
        "both_extensions_realized": bool(both_extensions_seen),
        "root_always_first": bool(root_always_first),
        "poset_reproduced": bool(all_realized_are_extensions
                                 and both_extensions_seen
                                 and root_always_first),
        "traj_A_first": (Tfast, Rfast),
        "traj_B_first": (Tslow, Rslow),
    }


def verify_motor_recovery_instantiation():
    """A second, NON-catatonia instantiation to show the machinery is general.
    Post-injury motor recovery as a branching identifiability graph:
        proximal (gross) -> distal (fine control)
        proximal (gross) -> timing (coordination)
    with distal and timing incomparable.  The theory predicts a recovery POSET:
    proximal first (Theorem 1), distal/timing in either order (Theorem 2), so the
    admissible orders are exactly the two linear extensions.  Offered as an
    illustrative, testable instantiation -- not a validated clinical result."""
    nodes = ["proximal", "distal", "timing"]
    edges = [("proximal", "distal"), ("proximal", "timing")]
    exts = set(linear_extensions(nodes, edges))
    realized = set()
    proximal_first = True
    for cd in (0.6, 1.0, 1.7):
        for ct in (0.6, 1.0, 1.7):
            order, _, _ = simulate_dag(nodes, edges, [1.0, cd, ct])
            realized.add(order)
            if order[0] != "proximal":
                proximal_first = False
    _, Td, Rd = simulate_dag(nodes, edges, [1.0, 1.7, 0.6])   # distal before timing
    _, Tt, Rt = simulate_dag(nodes, edges, [1.0, 0.6, 1.7])   # timing before distal
    return {
        "nodes": nodes, "edges": edges,
        "linear_extensions": sorted(exts),
        "realized_orders": sorted(realized),
        "all_realized_are_linear_extensions": bool(realized <= exts),
        "both_extensions_realized": bool(exts <= realized),
        "prerequisite_first": bool(proximal_first),
        "poset_reproduced": bool(realized <= exts and exts <= realized
                                 and proximal_first),
        "traj_distal_first": (Td, Rd),
        "traj_timing_first": (Tt, Rt),
    }


def verify_curriculum_instantiation():
    """A third instantiation -- deliberately NON-PROBABILISTIC -- showing the
    ordering law does not depend on the Fisher realization.

    Curriculum learning in an artificial agent.  A foundational skill S0 enables
    two intermediate skills S1, S2 that do not depend on each other, and both
    feed a composite skill S3 (a 'diamond' curriculum):

        S0 -> S1 -> S3
        S0 -> S2 -> S3          (S1, S2 incomparable)

    Here the recovery drive is the agent's per-skill LEARNING PROGRESS, gated by
    prerequisite mastery and decreasing in the remaining deficit.  That is exactly
    axioms (A3)-(A4) with NO likelihood, Fisher information or generative model:
    simulate_dag integrates the axiom-level law dr/dt = c * A * (1-r) directly, so
    it doubles as the non-probabilistic simulator.  The structure theorem then
    applies verbatim: the admissible acquisition orders are the linear extensions
    of the diamond (there are two -- S0 first, S3 last, S1/S2 interchange)."""
    nodes = ["S0", "S1", "S2", "S3"]
    edges = [("S0", "S1"), ("S0", "S2"), ("S1", "S3"), ("S2", "S3")]
    exts = set(linear_extensions(nodes, edges))
    realized = set()
    foundational_first = True
    composite_last = True
    for c1 in (0.6, 1.0, 1.7):
        for c2 in (0.6, 1.0, 1.7):
            order, _, _ = simulate_dag(nodes, edges, [1.0, c1, c2, 1.0])
            realized.add(order)
            if order[0] != "S0":
                foundational_first = False
            if order[-1] != "S3":
                composite_last = False
    _, T12, R12 = simulate_dag(nodes, edges, [1.0, 1.7, 0.6, 1.0])   # S1 before S2
    _, T21, R21 = simulate_dag(nodes, edges, [1.0, 0.6, 1.7, 1.0])   # S2 before S1
    return {
        "nodes": nodes, "edges": edges,
        "linear_extensions": sorted(exts),
        "realized_orders": sorted(realized),
        "all_realized_are_linear_extensions": bool(realized <= exts),
        "both_extensions_realized": bool(exts <= realized),
        "foundational_first": bool(foundational_first),
        "composite_last": bool(composite_last),
        "poset_reproduced": bool(realized <= exts and exts <= realized
                                 and foundational_first and composite_last),
        "traj_S1_first": (T12, R12),
        "traj_S2_first": (T21, R21),
    }


def verify_theorem_B_equivalence(s=S_SCALE):
    """Show the Fisher-first pipeline and the companion law give the SAME
    trajectory when c' = c/s, and both give the Saito Loop milestone order."""
    c = np.ones(N)                                  # per-channel Fisher constant
    r0 = np.full(N, 0.02)
    cprime = c / s
    Tf, Sf = simulate_fisher_first(c, r0, s=s, t_max=60.0, dt=0.02)
    Tc, Sc = simulate_companion(cprime, r0, t_max=60.0, dt=0.02)
    max_abs_diff = float(np.max(np.abs(Sf - Sc)))
    of, _ = milestone_order(Tf, Sf)
    oc, _ = milestone_order(Tc, Sc)
    return {
        "max_abs_trajectory_diff": max_abs_diff,
        "trajectories_coincide": bool(max_abs_diff < 1e-3),
        "fisher_first_order": of,
        "companion_order": oc,
        "both_are_saito_loop": bool(of == CLINICAL_SAITO_LOOP == oc),
        "T": Tf, "S_fisher": Sf, "S_companion": Sc,
    }


# =============================================================================
# 3d. PROPOSITION A.3 AND THEOREM 2(b): the gate family, and completeness
# =============================================================================
# Two checks that back the two claims Part A now proves in full generality.
#
# (A.3)  The ordering law uses the gate family only through (G1)-(G4) and the
#        support.  The proof of Theorem 1(b) needs only NESTEDNESS (G4),
#        phi_k <= phi_j whenever anc(k) contains anc(j) -- not the product form.
#        We therefore re-run the ordering test with a DIFFERENT gate of the same
#        support (the min gate, a t-norm) and require the same admissible orders.
#        The diamond is the smallest graph on which product and min differ, since
#        it is the smallest with a node having two ancestors.
#
# (Thm 2b) COMPLETENESS: every linear extension is realized.  Constructive: set
#        the rate constants to decay geometrically along a TARGET extension sigma.
#        Every edge runs forward in sigma, so c_k <= c_j holds automatically and
#        the construction never leaves the gating-dominated regime.  As the decay
#        ratio shrinks, the realized order must converge to sigma.

def _sim_dag_gate(nodes, edges, cprime, gate="product", r0=0.02, q=0.5,
                  t_max=None, dt=None):
    """Milestone order on an arbitrary DAG under an arbitrary gate family.
    The gate is evaluated as a masked reduction over the ancestor matrix, so the
    inner loop is vectorized; the horizon and step are scaled to the rate spread
    so that geometrically decaying rates stay affordable."""
    anc = _ancestors_from_edges(nodes, edges)
    idx = {n: i for i, n in enumerate(nodes)}
    m = len(nodes)
    M = np.zeros((m, m), dtype=bool)                 # M[k, j] = j is an ancestor of k
    for n in nodes:
        for p_ in anc[n]:
            M[idx[n], idx[p_]] = True
    c = np.asarray(cprime, float)
    if dt is None:
        dt = 0.02 / max(c.max(), 1.0)
    if t_max is None:
        t_max = 60.0 / c.min()                       # generous for the slowest capacity
    r = np.full(m, float(r0))
    T = np.full(m, np.inf)

    if gate == "product":
        def phi(rv):
            return np.prod(np.where(M, rv[None, :], 1.0), axis=1)
    else:                                            # min gate (a t-norm, same support)
        def phi(rv):
            return np.min(np.where(M, rv[None, :], 1.0), axis=1)

    def dr(rv):
        return c * phi(rv) * (1.0 - rv)

    n_steps = int(round(t_max / dt))
    for i in range(n_steps):
        k1 = dr(r)
        k2 = dr(np.clip(r + .5 * dt * k1, 0, 1))
        k3 = dr(np.clip(r + .5 * dt * k2, 0, 1))
        k4 = dr(np.clip(r + dt * k3, 0, 1))
        r = np.clip(r + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4), 0, 1)
        newly = (r >= q) & np.isinf(T)
        if newly.any():
            T[newly] = (i + 1) * dt
            if np.all(np.isfinite(T)):
                break
    return tuple(nodes[i] for i in np.argsort(T, kind="stable"))


def verify_gate_family_invariance(n_samples=200, seed=0):
    """Proposition A.3.  Diamond graph (the smallest with a two-ancestor node, so
    the product and min gates genuinely differ).  Under the rate condition, both
    gates must admit exactly the linear extensions, and nothing else."""
    nodes = ["S0", "S1", "S2", "S3"]
    edges = [("S0", "S1"), ("S0", "S2"), ("S1", "S3"), ("S2", "S3")]
    exts = set(linear_extensions(nodes, edges))
    rng = np.random.default_rng(seed)
    out = {}
    for gate in ("product", "min"):
        realized, viol = set(), 0
        for _ in range(n_samples):
            c0 = rng.uniform(1.0, 4.0)
            c1 = rng.uniform(0.2, c0)                      # c_k <= c_j on every edge
            c2 = rng.uniform(0.2, c0)
            c3 = rng.uniform(0.05, min(c1, c2))
            order = _sim_dag_gate(nodes, edges, [c0, c1, c2, c3], gate=gate)
            realized.add(order)
            if order not in exts:
                viol += 1
        out[gate] = {"samples": n_samples,
                     "order_violations": viol,
                     "realized_orders": sorted(realized),
                     "all_are_linear_extensions": viol == 0,
                     "both_extensions_realized": exts <= realized}
    out["same_admissible_set"] = bool(
        set(map(tuple, out["product"]["realized_orders"]))
        == set(map(tuple, out["min"]["realized_orders"])))
    out["linear_extensions"] = sorted(exts)
    return out


def verify_completeness(n_graphs=30, ratios=(0.9, 0.7, 0.5, 0.3), seed=1):
    """Theorem 2(b).  For a random DAG and a random TARGET linear extension sigma,
    set c_{sigma_i} = rho^(i-1).  Edges run forward in sigma, so the rate condition
    holds automatically.  The realized order must equal sigma once rho is small."""
    rng = np.random.default_rng(seed)
    cases = []
    for _ in range(n_graphs):
        m = int(rng.integers(4, 7))
        nodes = ["n%d" % i for i in range(m)]
        edges = [(nodes[i], nodes[j])
                 for i in range(m) for j in range(i + 1, m) if rng.random() < 0.35]
        exts = linear_extensions(nodes, edges)
        cases.append((nodes, edges, exts[int(rng.integers(len(exts)))]))
    out = {}
    for rho in ratios:
        hits = 0
        for nodes, edges, sigma in cases:
            c = {q: rho ** i for i, q in enumerate(sigma)}
            realized = _sim_dag_gate(nodes, edges, [c[q] for q in nodes])
            hits += (realized == tuple(sigma))
        out[str(rho)] = {"targets": len(cases), "realized": int(hits)}
    out["complete_at_small_ratio"] = bool(
        out[str(min(ratios))]["realized"] == len(cases))
    return out


# =============================================================================
# 4. Pre-declared robustness grid through the Fisher-first pipeline
# =============================================================================
def robustness_grid_fisher_first():
    """Pre-declared grid over nuisance hyperparameters; every point is driven
    through the Fisher-first pipeline dQ/dt = c * prod r.  Returns the set of
    admitted milestone orderings (should be exactly {Saito Loop})."""
    scales = (0.6, 1.0, 1.6)          # global Fisher-constant scale
    thetas_ms = (0.3, 0.5, 0.7)       # milestone threshold
    onsets = (0.01, 0.02, 0.05)       # uniform onset
    ss = (0.5, 1.0, 2.0)              # link scale s
    seeds = (0, 1, 2)                 # +/-5% per-channel jitter
    admitted = {}
    n_points = 0
    for g in scales:
        for theta in thetas_ms:
            for r0v in onsets:
                for s in ss:
                    for seed in seeds:
                        rng = np.random.default_rng(seed)
                        jit = 1.0 + 0.05 * (rng.random(N) - 0.5) * 2
                        c = g * jit
                        r0 = np.full(N, r0v)
                        T, S = simulate_fisher_first(c, r0, s=s,
                                                     t_max=240.0, dt=0.05)
                        order, _ = milestone_order(T, S, theta=theta)
                        admitted[order] = admitted.get(order, 0) + 1
                        n_points += 1
    return {"n_points": n_points, "admitted": admitted,
            "n_admitted": len(admitted),
            "n_permutations_total": 120}


# =============================================================================
# 5. FIGURES
# =============================================================================
plt.rcParams.update({
    "font.family": "DejaVu Serif", "font.size": 10, "axes.linewidth": 0.8,
    "savefig.dpi": 300, "figure.dpi": 150, "mathtext.fontset": "dejavuserif",
})
_COLORS = ["#1b4965", "#2a9d8f", "#e9c46a", "#e76f51", "#9b2226"]
_CMAP = {q: _COLORS[i] for i, q in enumerate(QUANTITIES)}


def figure1_factorization(resA, path):
    """Fig 1. (a) Fisher information equals c_k * prod r_j: Monte-Carlo estimate
    vs closed form on the identity line.  (b) support of the Fisher field
    reproduces the prerequisite graph (zero-pattern reachability)."""
    from matplotlib.colors import ListedColormap
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.2, 3.7),
                                   gridspec_kw={"width_ratios": [1.15, 1.0]})
    pts = np.array(resA["scatter_points"])
    lo, hi = 0.0, pts.max() * 1.05
    axA.plot([lo, hi], [lo, hi], color="#adb5bd", lw=1.0, ls="--", zorder=1)
    axA.scatter(pts[:, 0], pts[:, 1], s=16, color="#2a9d8f",
                edgecolors="black", linewidths=0.3, zorder=3)
    axA.set_xlim(lo, hi); axA.set_ylim(lo, hi)
    axA.set_xlabel(r"closed form  $c_k\,\prod_{j\in\mathrm{anc}(k)} r_j$")
    axA.set_ylabel(r"Monte-Carlo  $\mathrm{Var}(\mathrm{score})$")
    axA.set_title("a  Fisher information factorizes (Theorem A)",
                  loc="left", fontsize=10, fontweight="bold")
    axA.text(0.05 * hi, 0.9 * hi,
             f"max rel. error\n{resA['max_rel_err_montecarlo']*100:.1f}%",
             fontsize=8, color="#495057")

    R = resA["support_matrix"]
    order = QUANTITIES
    axB.imshow(R, cmap=ListedColormap(["#f1f3f5", "#1b4965"]),
               vmin=0, vmax=1, aspect="equal")
    axB.set_xticks(range(N)); axB.set_yticks(range(N))
    axB.set_xticklabels([LABELS[q] for q in order])
    axB.set_yticklabels([LABELS[q] for q in order])
    axB.set_xlabel(r"makes $I_k$ vanish")
    axB.set_ylabel(r"zeroing $r_i$")
    for i in range(N):
        for j in range(N):
            if R[i, j]:
                axB.text(j, i, "1", ha="center", va="center",
                         color="white", fontsize=9)
    axB.set_title("b  Support of $I_k$ = prerequisite graph",
                  loc="left", fontsize=10, fontweight="bold")
    _finalize(fig, path)


def figure2_accumulation_and_law(resB, accB, path):
    """Fig 2. (a) dP/dt = I: exact conjugate precision addition vs accumulation.
    (b) Fisher-first recovery r_k(t) (solid) coincides with the companion law
    (dashed); milestones cross in Saito Loop order."""
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.4, 3.8))

    # (a) precision accumulation
    theta, gate = 1.0, 0.6
    c = per_informative_fisher(theta)
    rng = np.random.default_rng(3)
    n = 6000
    informative = rng.random(n) < gate
    P0 = 1.0
    P_exact = P0 + c * np.cumsum(informative)
    P_ode = P0 + c * gate * np.arange(1, n + 1)
    x = np.arange(1, n + 1)
    axA.plot(x, P_exact, color="#1b4965", lw=1.4,
             label="exact conjugate addition")
    axA.plot(x, P_ode, color="#e76f51", lw=1.4, ls="--",
             label=r"$dP_k/dt = I_k$")
    axA.set_xlabel("observations (arb. time)")
    axA.set_ylabel(r"posterior precision  $P_k$")
    axA.set_title("a  Precision accumulates Fisher information",
                  loc="left", fontsize=10, fontweight="bold")
    axA.legend(loc="upper left", frameon=False, fontsize=8)

    # (b) trajectories coincide + milestones
    T, Sf, Sc = resB["T"], resB["S_fisher"], resB["S_companion"]
    theta_ms = 0.5
    times = milestone_times(T, Sf, theta=theta_ms)
    for k, q in enumerate(QUANTITIES):
        axB.plot(T, Sf[:, k], color=_CMAP[q], lw=2.0, label=LABELS[q])
        axB.plot(T, Sc[:, k], color="black", lw=0.8, ls=(0, (1, 1)), alpha=0.7)
        if np.isfinite(times[k]):
            axB.plot([times[k]], [theta_ms], "o", color=_CMAP[q],
                     mec="black", mew=0.7, ms=7, zorder=5)
    axB.axhline(theta_ms, color="grey", ls="--", lw=0.8)
    axB.set_xlim(0, 12)
    axB.set_xlabel("time (arb. units)")
    axB.set_ylabel(r"recovery state  $r_k(t)$"); axB.set_ylim(0, 1.02)
    axB.set_title("b  Derived law = companion law → Saito Loop",
                  loc="left", fontsize=10, fontweight="bold")
    axB.legend(loc="lower right", frameon=False, fontsize=8, ncol=1)
    axB.text(0.4, 0.92,
             f"max |Δ trajectory| = {resB['max_abs_trajectory_diff']:.1e}",
             fontsize=8, color="#495057")
    _finalize(fig, path)


def figureS1_saturation(path):
    """Suppl. Fig. 1 | The derived saturating factor. (a) link r = 1 - exp(-Q/s) and its
    deficit-proportional slope; (b) marginal gain dr/dQ vs r is the straight
    line (1-r)/s, versus the (1-r)^2 slope of the variance normalization."""
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.2, 3.7))
    Q = np.linspace(0, 5, 400)
    for s, col in [(0.5, "#1b4965"), (1.0, "#2a9d8f"), (2.0, "#e76f51")]:
        axA.plot(Q, 1 - np.exp(-Q / s), color=col, lw=1.8, label=f"$s={s}$")
    axA.set_xlabel(r"accumulated information  $Q_k = P_k - P_0$")
    axA.set_ylabel(r"recovered fraction  $r_k = 1-e^{-Q_k/s}$")
    axA.set_ylim(0, 1.02)
    axA.set_title("a  Deficit-proportional (relaxation) link",
                  loc="left", fontsize=10, fontweight="bold")
    axA.legend(loc="lower right", frameon=False, fontsize=8)

    r = np.linspace(0, 1, 400)
    axB.plot(r, (1 - r), color="#2a9d8f", lw=2.0,
             label=r"exponential link:  $dr/dQ \propto (1-r)$")
    axB.plot(r, (1 - r) ** 2, color="#e76f51", lw=2.0, ls="--",
             label=r"variance link:  $dr/dQ \propto (1-r)^2$")
    axB.set_xlabel(r"recovered fraction  $r_k$")
    axB.set_ylabel(r"marginal gain per unit information")
    axB.set_title("b  The derived factor $(1-r_k)$",
                  loc="left", fontsize=10, fontweight="bold")
    axB.legend(loc="upper right", frameon=False, fontsize=8)
    _finalize(fig, path)


def figureS4_curriculum(cur, path):
    """Suppl. Fig. 4. A NON-probabilistic instantiation: curriculum learning as a
    diamond dependency graph S0 -> {S1, S2} -> S3 -> a recovery poset.  The
    foundational skill is acquired first and the composite last; the two
    intermediate skills swap with their learning rates (the two linear
    extensions).  No Fisher information or likelihood enters."""
    nodes = cur["nodes"]
    colors = {"S0": "#1b4965", "S1": "#2a9d8f", "S2": "#e76f51", "S3": "#9b5de5"}
    labels = {"S0": "S0 (foundational)", "S1": "S1 (intermediate)",
              "S2": "S2 (intermediate)", "S3": "S3 (composite)"}
    fig, axes = plt.subplots(1, 2, figsize=(7.3, 3.0), sharey=True)
    for ax, key, title in (
        (axes[0], "traj_S1_first",
         "one regime:  S0 $\\rightarrow$ S1 $\\rightarrow$ S2 $\\rightarrow$ S3"),
        (axes[1], "traj_S2_first",
         "another regime:  S0 $\\rightarrow$ S2 $\\rightarrow$ S1 $\\rightarrow$ S3"),
    ):
        Tt, Rr = cur[key]
        for i, q in enumerate(nodes):
            ax.plot(Tt, Rr[:, i], color=colors[q], lw=2.2, label=labels[q])
        ax.axhline(0.5, ls=":", color="0.5", lw=1)
        ax.set_xlabel("training time (arb. units)")
        ax.set_title(title, fontsize=9.5)
        ax.set_xlim(0, 20); ax.set_ylim(0, 1.02)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("skill mastery  r(t)")
    axes[0].legend(frameon=False, fontsize=8.0, loc="lower right")
    fig.suptitle("Curriculum learning (no likelihood): foundational skill first, "
                 "composite last; intermediates swap (the two linear extensions)",
                 fontsize=9.5, y=1.02)
    _finalize(fig, path, dpi=200)


def figureS3_motor(mot, path):
    """Suppl. Fig. 3 | Instantiation 2: post-injury motor recovery as a branching
    dependency graph -> a recovery poset (the two linear extensions)."""
    nodes = mot["nodes"]
    colors = {"proximal": "#1b4965", "distal": "#2a9d8f", "timing": "#e76f51"}
    labels = {"proximal": "proximal (gross)", "distal": "distal (fine)",
              "timing": "timing (coord.)"}
    fig, axes = plt.subplots(1, 2, figsize=(7.3, 3.0), sharey=True)
    for ax, key, title in (
        (axes[0], "traj_distal_first",
         "one regime:  proximal $\\rightarrow$ distal $\\rightarrow$ timing"),
        (axes[1], "traj_timing_first",
         "another regime:  proximal $\\rightarrow$ timing $\\rightarrow$ distal"),
    ):
        Tt, Rr = mot[key]
        for i, q in enumerate(nodes):
            ax.plot(Tt, Rr[:, i], color=colors[q], lw=2.2, label=labels[q])
        ax.axhline(0.5, ls=":", color="0.5", lw=1)
        ax.set_xlabel("time (arb. units)")
        ax.set_title(title, fontsize=9.5)
        ax.set_xlim(0, Tt[-1]); ax.set_ylim(0, 1.02)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("recovery state  r(t)")
    axes[0].legend(frameon=False, fontsize=8.5, loc="lower right")
    fig.suptitle("Motor recovery instantiation: proximal base recovers first; "
                 "fine control and timing swap (the two linear extensions)",
                 fontsize=10, y=1.02)
    _finalize(fig, path, dpi=200)




# =============================================================================
# 3c. CORRECTED ORDERING THEOREM: onset vs milestone precedence and its regime
# =============================================================================
# Peer review established (and this section reproduces) that the *unconditional*
# form of Theorem 1 is false: with a common onset r0 > 0 a descendant whose rate
# constant far exceeds its prerequisite's can reach a high milestone FIRST, even
# though every axiom (A1-A4) and Proposition A.3 hold.  The corrected theorem is
#   (a) ONSET precedence  : unconditional (gate vanishes at baseline, r0 = 0);
#   (b) MILESTONE precedence: holds in the gating-dominated regime, whose clean
#       sufficient condition is that rate constants do NOT increase along edges
#       (c_k <= c_j for every edge j->k).
# The functions below (i) exhibit the counterexample, (ii) confirm the sufficient
# condition on a 5-node chain, (iii) confirm onset precedence is unconditional,
# and (iv) map the reversal boundary c_k/c_j vs the onset r0 (-> Figure 5).

def _chain_gate(R):
    """Product gate of a path chain, vectorized over a batch of trajectories.
    R has shape (B, m); returns g with g[:, k] = prod_{j < k} R[:, j] (unity at k = 0).
    Identical, multiplication-order for multiplication-order, to the scalar loop."""
    g = np.ones_like(R)
    if R.shape[1] > 1:
        g[:, 1:] = np.cumprod(R, axis=1)[:, :-1]
    return g


def _milestone_times_chain_batch(C, r0, q, t_max, dt):
    """Milestone times T_k(q) for a BATCH of path chains, integrated with the same
    fixed-step RK4 and the same product gate / deficit closure as _sim_chain_generic.
    C: (B, m) rate constants.  r0: scalar or (B,) common onset.  Returns (B, m).

    This is a pure speed-up: the arithmetic is step-for-step identical to running
    _sim_chain_generic once per row, but the history is never stored, so the whole
    sweep runs as B trajectories in parallel instead of B Python loops."""
    C = np.asarray(C, float)
    B, m = C.shape
    r0 = np.broadcast_to(np.asarray(r0, float), (B,)).astype(float)
    R = np.repeat(r0[:, None], m, axis=1)
    T = np.where(R >= q, 0.0, np.inf)

    def dr(Rv):
        return C * _chain_gate(Rv) * (1.0 - Rv)

    for i in range(int(round(t_max / dt))):
        k1 = dr(R)
        k2 = dr(R + 0.5 * dt * k1)
        k3 = dr(R + 0.5 * dt * k2)
        k4 = dr(R + dt * k3)
        R = R + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        newly = (R >= q) & np.isinf(T)
        if newly.any():
            T[newly] = (i + 1) * dt
    return T


def _sim_chain_generic(c, r0, t_max=90.0, dt=5e-3):
    """RK4 for dr_k/dt = c_k * (prod_{j<k} r_j) * (1 - r_k) on a path chain
    1 -> 2 -> ... -> m (product gate, deficit-closing closure).  Vectorized gate."""
    c = np.asarray(c, float); m = len(c)
    r = np.full(m, float(r0)); n = int(round(t_max / dt))
    ts = np.zeros(n + 1); H = np.zeros((n + 1, m)); H[0] = r

    def dr(rv):
        return c * _chain_gate(rv[None, :])[0] * (1.0 - rv)

    for i in range(n):
        k1 = dr(r); k2 = dr(r + .5 * dt * k1)
        k3 = dr(r + .5 * dt * k2); k4 = dr(r + dt * k3)
        r = r + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        H[i + 1] = r; ts[i + 1] = (i + 1) * dt
    return ts, H


def _milestone_time_1d(ts, x, q):
    j = int(np.argmax(x >= q))
    return ts[j] if x[j] >= q else np.inf


def verify_prerequisite_precedence_counterexample():
    """Edge 1->2, common onset r0=0.5, c1=1, c2=100 (product gate, (1-r) closure).
    Every axiom A1-A4 and Proposition A.3 hold, yet the child crosses q=0.9 long
    before the parent, so the UNCONDITIONAL form of Theorem 1 is false."""
    c1, c2, r0, q = 1.0, 100.0, 0.5, 0.9
    ts, H = _sim_chain_generic([c1, c2], r0, t_max=40.0, dt=1e-4)
    T1 = _milestone_time_1d(ts, H[:, 0], q)
    T2 = _milestone_time_1d(ts, H[:, 1], q)
    dprime0 = (1.0 - r0) * (c1 - c2 * r0)      # sign of d(delta)/dt at t=0
    return {
        "r0": r0, "c_parent": c1, "c_child": c2, "q": q,
        "T_parent": float(T1), "T_child": float(T2),
        "delta_prime_0": float(dprime0),
        "child_reaches_milestone_first": bool(T2 < T1),
        "axioms_A1_A4_and_PropA3_satisfied": True,   # product gate + (1-r), c>0, r0 in [0,1)
    }


def verify_rate_compatibility_condition(n_samples=36, seed=0):
    """5-node chain.  Sufficient condition of Theorem 1(b): rate constants
    non-increasing along edges (c_k <= c_j).  Compared against the complementary
    case (rates increasing along edges).  The 108 trajectories of each arm are
    integrated as one batch; the random draws keep their original order, so the
    numbers are identical to the scalar version."""
    rng = np.random.default_rng(seed)
    out = {}
    for label, incr in (("non_increasing", False), ("increasing", True)):
        rows_c, rows_r0 = [], []
        for _ in range(n_samples):
            s_ = np.sort(rng.uniform(0.3, 4.0, 5))
            base = s_ if incr else s_[::-1]
            for r0 in (0.0, 0.3, 0.6):
                rows_c.append(np.asarray(base, float))
                rows_r0.append(r0)
        T = _milestone_times_chain_batch(np.array(rows_c), np.array(rows_r0),
                                         q=0.9, t_max=80.0, dt=7e-3)
        viol = 0
        for row in T:
            if list(np.argsort(row, kind="stable")) != [0, 1, 2, 3, 4]:
                viol += 1                    # chain: the only linear extension is the identity
        out[label] = {"samples": int(T.shape[0]), "graph_order_violations": int(viol)}
    return out


def verify_onset_precedence(n_samples=120, seed=1):
    """5-node chain, r0 -> 0.  The departure-from-baseline order is a linear
    extension for ARBITRARY (even strongly increasing) rate constants."""
    rng = np.random.default_rng(seed)
    C = np.array([rng.uniform(0.3, 20.0, 5) for _ in range(n_samples)])
    T = _milestone_times_chain_batch(C, 1e-9, q=1e-6, t_max=40.0, dt=3e-3)
    bad = sum(1 for row in T
              if list(np.argsort(row, kind="stable")) != [0, 1, 2, 3, 4])
    return {"samples": int(n_samples), "onset_order_violations": int(bad)}


def reversal_boundary(onsets=(0.0, 0.1, 0.3, 0.5, 0.7)):
    """Bisection for the rate ratio c_child/c_parent at which the milestone order
    of a single edge reverses, as a function of the common onset r0 (q = 0.9).
    All onsets are bisected simultaneously."""
    r0s = np.array([max(r0, 1e-9) for r0 in onsets], float)

    def reverses(ratios):
        C = np.stack([np.ones_like(ratios), ratios], axis=1)
        T = _milestone_times_chain_batch(C, r0s, q=0.9, t_max=35.0, dt=2e-3)
        return T[:, 1] < T[:, 0]

    lo = np.ones(len(onsets)); hi = np.full(len(onsets), 6.0)
    reversible = reverses(hi)
    for _ in range(16):
        mid = 0.5 * (lo + hi)
        rev = reverses(mid)
        hi = np.where(rev, mid, hi)
        lo = np.where(rev, lo, mid)
    mids = 0.5 * (lo + hi)
    return {r0: (float(mids[i]) if reversible[i] else float("inf"))
            for i, r0 in enumerate(onsets)}


def gating_phase_map(ratios=None, r0s=None, q=0.9):
    """Milestone order over a single edge as a function of the rate ratio c_k/c_j
    and the common onset r0.  Z = 1 where the prerequisite completes first (graph
    order preserved), 0 where the descendant overtakes it.  Used by Fig. 3c.
    The whole (r0 x ratio) grid is integrated as a single batch."""
    ratios = np.linspace(0.5, 4.0, 30) if ratios is None else np.asarray(ratios)
    r0s = np.linspace(0.0, 0.85, 26) if r0s is None else np.asarray(r0s)
    RA, R0 = np.meshgrid(ratios, r0s)                  # (n_r0, n_ratio)
    flat_ratio = RA.ravel()
    flat_r0 = np.maximum(R0.ravel(), 1e-9)
    C = np.stack([np.ones_like(flat_ratio), flat_ratio], axis=1)
    T = _milestone_times_chain_batch(C, flat_r0, q=q, t_max=22.0, dt=3e-3)
    Z = (T[:, 0] <= T[:, 1]).astype(float).reshape(RA.shape)
    return ratios, r0s, Z


def figure3_ordering(grid, dag, path, phase=None):
    """Fig. 3 | Admissible orders are the linear extensions of the graph -- and
    where the law ends.
    (a) chain: one milestone ordering at every point of a pre-declared grid.
    (b) branch: both linear extensions occur; the prerequisite is never overtaken.
    (c) boundary of prerequisite-gating dominance in the (rate-ratio, onset) plane.
    """
    from matplotlib.colors import ListedColormap
    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(11.4, 3.25),
                                        gridspec_kw={"width_ratios": [1.0, 1.05, 1.05]})

    # ---- (a) chain: a single admitted order across the grid ------------------
    n_pts = grid["n_points"]
    rows = int(np.floor(np.sqrt(n_pts)))
    while rows > 1 and n_pts % rows != 0:
        rows -= 1
    cols = n_pts // rows
    axA.imshow(np.ones((rows, cols)), cmap=ListedColormap(["#2a9d8f"]),
               vmin=0, vmax=1, aspect="auto")
    for k in range(cols + 1):
        axA.axvline(k - 0.5, color="white", lw=0.4)
    for k in range(rows + 1):
        axA.axhline(k - 0.5, color="white", lw=0.4)
    axA.set_xticks([]); axA.set_yticks([])
    axA.text(cols / 2 - 0.5, rows / 2 - 0.5,
             r"$\pi_s \rightarrow \beta \rightarrow \pi_m \rightarrow "
             r"\pi_{v\_\mathrm{fast}} \rightarrow \pi_{v\_\mathrm{slow}}$"
             "\n\n"
             f"{grid['n_admitted']} of {grid['n_permutations_total']}\norderings admitted",
             ha="center", va="center", color="white", fontsize=10, fontweight="bold")
    axA.set_title("a  chain " + r"$\rightarrow$" + " a unique order",
                  loc="left", fontsize=10, fontweight="bold")
    axA.set_xlabel(f"{n_pts} grid points\n(scale x threshold x onset x s x seed)",
                   fontsize=8)

    # ---- (b) branch: a recovery poset ---------------------------------------
    nodes = dag["nodes"]
    colors = {"root": "#1b4965", "A": "#2a9d8f", "B": "#e76f51"}
    regimes = [("regime 1\n" + r"$c_A > c_B$", [1.0, 1.6, 0.6]),
               ("regime 2\n" + r"$c_B > c_A$", [1.0, 0.6, 1.6])]
    for row, (lab, cp) in enumerate(regimes):
        order, Tt, Rr = simulate_dag(nodes, dag["edges"], cp)
        y = 1 - row
        for i, q in enumerate(nodes):
            j = int(np.argmax(Rr[:, i] >= 0.5))
            tk = Tt[j]
            axB.plot([tk], [y], "o", ms=11, color=colors[q], mec="black", mew=0.6,
                     zorder=3)
            dy = -22 if q == "root" else 13
            axB.annotate(q, (tk, y), xytext=(0, dy), textcoords="offset points",
                         ha="center", fontsize=8.5, color=colors[q],
                         fontweight="bold")
        axB.plot([0.35, 2.6], [y, y], color="0.85", lw=1.0, zorder=1)
        axB.text(0.42, y + 0.30, lab, ha="left", va="center", fontsize=8)
    axB.set_yticks([]); axB.set_ylim(-0.85, 1.85); axB.set_xlim(0.35, 2.6)
    axB.set_xlabel(r"milestone time  $T_k(q=0.5)$")
    axB.set_title("b  branch " + r"$\rightarrow$" + " a recovery poset",
                  loc="left", fontsize=10, fontweight="bold")
    axB.text(1.5, -0.66, "root never overtaken; A and B swap\n(the two linear extensions)",
             ha="center", va="center", fontsize=7.6, color="#495057")
    axB.spines[["top", "right", "left"]].set_visible(False)

    # ---- (c) boundary of gating dominance -----------------------------------
    ratios, r0s, Z = gating_phase_map() if phase is None else phase
    axC.pcolormesh(ratios, r0s, Z, shading="auto", vmin=0, vmax=1,
                   cmap=matplotlib.colors.ListedColormap([_COLORS[4], _COLORS[1]]))
    bnd = []
    for i in range(len(r0s)):
        row = Z[i]; k = int(np.argmax(row < 0.5))
        bnd.append(ratios[k] if row[k] < 0.5 else np.nan)
    axC.plot(bnd, r0s, color="k", lw=1.5)
    axC.axvline(1.0, ls="--", color="w", lw=1.0)
    axC.text(0.58, 0.70, "prerequisite first\n(graph order)", color="w",
             fontsize=7.2, ha="left", va="center")
    axC.text(2.75, 0.68, "descendant first\n(reversal)", color="w",
             fontsize=7.2, ha="center", va="center")
    axC.set_xlabel(r"edge rate ratio  $c_k/c_j$")
    axC.set_ylabel(r"common onset  $r_0$")
    axC.set_title("c  boundary of gating dominance",
                  loc="left", fontsize=10, fontweight="bold")
    axC.set_xlim(0.5, 4.0); axC.set_ylim(0, 0.85)
    _finalize(fig, path)


def figureS2_counterexample(path):
    """Suppl. Fig. 2 | Sharpness of the rate condition: a fast descendant
    overtakes its prerequisite (r0 = 0.5, c_j = 1, c_k = 100, q = 0.9)."""
    fig, ax = plt.subplots(figsize=(4.6, 3.1))
    ts, H = _sim_chain_generic([1.0, 100.0], 0.5, t_max=4.0, dt=1e-4)
    q = 0.9
    ax.plot(ts, H[:, 0], color=_COLORS[0], lw=1.9,
            label=r"prerequisite $r_j$  ($c_j = 1$)")
    ax.plot(ts, H[:, 1], color=_COLORS[4], lw=1.9,
            label=r"descendant $r_k$  ($c_k = 100$)")
    ax.axhline(q, ls=":", color="0.5", lw=0.9)
    T1 = _milestone_time_1d(ts, H[:, 0], q)
    T2 = _milestone_time_1d(ts, H[:, 1], q)
    ax.plot([T1], [q], "o", color=_COLORS[0], ms=5.5, mec="k", mew=0.5)
    ax.plot([T2], [q], "o", color=_COLORS[4], ms=5.5, mec="k", mew=0.5)
    ax.annotate(r"$T_k$", (T2, q), xytext=(T2 + 0.22, q - 0.16), color=_COLORS[4])
    ax.annotate(r"$T_j$", (T1, q), xytext=(T1 - 0.12, q - 0.16), color=_COLORS[0],
                ha="right")
    ax.set_xlim(0, 4); ax.set_ylim(0.45, 1.02)
    ax.set_xlabel("time (arb. units)")
    ax.set_ylabel(r"recovered fraction  $r_k$")
    ax.legend(frameon=False, fontsize=8, loc="center right")
    ax.spines[["top", "right"]].set_visible(False)
    _finalize(fig, path, dpi=300)


# =============================================================================
# 6. DRIVER
# =============================================================================

def main(fig_dir="."):
    import functools, time
    _t0 = time.time()
    print_ = functools.partial(print, flush=True)   # never look hung on a slow box
    os.makedirs(fig_dir, exist_ok=True)   # create the target dir if needed
    report = {}
    print_("=" * 74)
    print_("SAITO LOOP, information-geometric derivation: numerical verification")
    print_("=" * 74)

    print_("\n[1/8] Theorem A: Fisher factorization (Monte-Carlo, ~5 s) ...")
    # Theorem A
    resA = verify_theorem_A()
    print_("\n[Theorem A]  Fisher information factorizes as c_k * prod r_j")
    print_(f"  channels x levels checked        : {resA['checked']}")
    print_(f"  closed-form == c_k*prod r (exact): max rel err "
          f"{resA['max_rel_err_analytic_vs_product']:.2e}")
    print_(f"  Monte-Carlo == closed-form       : max rel err "
          f"{resA['max_rel_err_montecarlo']*100:.2f}%")
    print_(f"  support(I_k) == prerequisite graph: "
          f"{resA['support_reproduces_prerequisite_graph']}")
    report["theoremA"] = {
        "checked": resA["checked"],
        "max_rel_err_analytic_vs_product": resA["max_rel_err_analytic_vs_product"],
        "max_rel_err_montecarlo": resA["max_rel_err_montecarlo"],
        "support_reproduces_prerequisite_graph":
            resA["support_reproduces_prerequisite_graph"],
    }

    print_("\n[2/8] Theorem B1 ...")
    # Theorem B (B1)
    accB = verify_precision_accumulation()
    print_("\n[Theorem B1] posterior precision accumulates Fisher info: dP/dt = I")
    print_(f"  exact conjugate P_final : {accB['P_exact_final']:.2f}")
    print_(f"  ODE dP/dt=I    P_final  : {accB['P_ode_final']:.2f}")
    print_(f"  relative discrepancy    : {accB['rel_discrepancy_final']*100:.2f}% "
          f"-> matches = {accB['matches']}")
    report["theoremB1"] = accB

    print_("\n[3/8] Theorem B2/B3 ...")
    # Theorem B (B2/B3)
    resB = verify_theorem_B_equivalence()
    print_("\n[Theorem B2/B3] derived law reproduces the companion recovery law")
    print_(f"  max |trajectory difference| : {resB['max_abs_trajectory_diff']:.2e}"
          f" -> coincide = {resB['trajectories_coincide']}")
    print_(f"  Fisher-first milestone order: "
          f"{' -> '.join(resB['fisher_first_order'])}")
    print_(f"  companion milestone order   : "
          f"{' -> '.join(resB['companion_order'])}")
    print_(f"  both are the Saito Loop     : {resB['both_are_saito_loop']}")
    report["theoremB23"] = {
        "max_abs_trajectory_diff": resB["max_abs_trajectory_diff"],
        "trajectories_coincide": resB["trajectories_coincide"],
        "both_are_saito_loop": resB["both_are_saito_loop"],
    }

    print_("\n[4/8] Robustness grid, 243 points (~20 s) ...")
    # Robustness grid through the Fisher-first pipeline
    grid = robustness_grid_fisher_first()
    only = list(grid["admitted"].keys())
    print_("\n[Robustness] Fisher-first pipeline over a pre-declared grid")
    print_(f"  points evaluated    : {grid['n_points']}")
    print_(f"  distinct orderings  : {grid['n_admitted']} of "
          f"{grid['n_permutations_total']}")
    if grid["n_admitted"] == 1:
        print_(f"  admitted ordering   : {' -> '.join(only[0])}")
    report["robustness"] = {"n_points": grid["n_points"],
                            "n_admitted": grid["n_admitted"]}

    print_("\n[5/8] Ordering theorem: counterexample, rate condition, onset, boundary (~15 s) ...")
    # Corrected ordering theorem: onset (unconditional) vs milestone (regime)
    print_("\n[Theorem 1 corrected] onset precedence vs milestone precedence")
    ce = verify_prerequisite_precedence_counterexample()
    print_(f"  counterexample r0={ce['r0']}, c_parent={ce['c_parent']}, "
          f"c_child={ce['c_child']} (q={ce['q']}):")
    print_(f"    T_parent={ce['T_parent']:.3f}  T_child={ce['T_child']:.3f}  "
          f"child first = {ce['child_reaches_milestone_first']}")
    print_(f"    axioms A1-A4 & Prop A.3 all hold = "
          f"{ce['axioms_A1_A4_and_PropA3_satisfied']}  "
          f"=> unconditional Theorem 1 is FALSE")
    rc = verify_rate_compatibility_condition()
    print_(f"  sufficient condition c_k<=c_j (5-node chain): "
          f"non-increasing {rc['non_increasing']['graph_order_violations']}/"
          f"{rc['non_increasing']['samples']} violations; "
          f"increasing {rc['increasing']['graph_order_violations']}/"
          f"{rc['increasing']['samples']} violations")
    op = verify_onset_precedence()
    print_(f"  onset precedence (arbitrary rates): "
          f"{op['onset_order_violations']}/{op['samples']} violations "
          f"(0 => unconditional)")
    bnd = reversal_boundary()
    bnd_str = ", ".join(f"r0={k:.1f}:{('%.2f'%v) if v!=float('inf') else 'none'}"
                        for k, v in bnd.items())
    print_(f"  reversal boundary c_child/c_parent : {bnd_str}")
    report["theorem1_corrected"] = {
        "counterexample": ce,
        "rate_condition": rc,
        "onset_precedence": op,
        "reversal_boundary": {str(k): v for k, v in bnd.items()},
    }

    print_("\n[6/8] Branching DAG and the two instantiations ...")
    # Proposition A.3: the gate enters only through its support
    print_("\n[5b/8] Proposition A.3 and Theorem 2(b) ...")
    gf = verify_gate_family_invariance()
    print_("\n[Proposition A.3] the gate family enters only through its support")
    print_(f"  diamond, rate condition enforced; product gate: "
           f"{gf['product']['order_violations']}/{gf['product']['samples']} violations")
    print_(f"  same graph, MIN gate (same support, a t-norm)  : "
           f"{gf['min']['order_violations']}/{gf['min']['samples']} violations")
    print_(f"  both gates admit the SAME set of orders        : {gf['same_admissible_set']}")
    print_(f"  and that set is the linear extensions          : "
           f"{gf['product']['all_are_linear_extensions'] and gf['min']['all_are_linear_extensions']}")
    report["propositionA3"] = {"same_admissible_set": gf["same_admissible_set"],
                               "product_violations": gf["product"]["order_violations"],
                               "min_violations": gf["min"]["order_violations"]}

    comp = verify_completeness()
    print_("\n[Theorem 2(b)] completeness: every linear extension is realized")
    print_("  random DAGs, rates decaying along a target extension sigma:")
    for rho, d in comp.items():
        if rho == "complete_at_small_ratio":
            continue
        print_(f"    decay ratio {rho}: {d['realized']}/{d['targets']} targets realized")
    print_(f"  all targets realized at the smallest ratio      : "
           f"{comp['complete_at_small_ratio']}")
    report["theorem2b_completeness"] = comp

    # Theorem 2: a branching DAG yields a recovery poset (linear extensions)
    dag = verify_partial_order_dag()
    print_("\n[Theorems 1-3] branching DAG root->A, root->B (A,B incomparable)")
    print_(f"  linear extensions of graph  : "
          f"{[' -> '.join(e) for e in dag['linear_extensions']]}")
    print_(f"  realized recovery orders    : "
          f"{[' -> '.join(o) for o in dag['realized_orders']]}")
    print_(f"  every order is an extension  : "
          f"{dag['all_realized_are_linear_extensions']}")
    print_(f"  both extensions realized     : {dag['both_extensions_realized']}")
    print_(f"  prerequisite always first    : {dag['root_always_first']}")
    report["theorems123_poset"] = {
        "linear_extensions": dag["linear_extensions"],
        "realized_orders": dag["realized_orders"],
        "poset_reproduced": dag["poset_reproduced"],
    }

    # Second instantiation beyond catatonia: motor recovery (branching graph)
    mot = verify_motor_recovery_instantiation()
    print_("\n[Second instantiation] motor recovery: proximal -> {distal, timing}")
    print_(f"  linear extensions           : "
          f"{[' -> '.join(e) for e in mot['linear_extensions']]}")
    print_(f"  realized recovery orders    : "
          f"{[' -> '.join(o) for o in mot['realized_orders']]}")
    print_(f"  every order is an extension  : "
          f"{mot['all_realized_are_linear_extensions']}")
    print_(f"  both extensions realized     : {mot['both_extensions_realized']}")
    print_(f"  prerequisite (proximal) first: {mot['prerequisite_first']}")
    report["motor_instantiation"] = {
        "linear_extensions": mot["linear_extensions"],
        "realized_orders": mot["realized_orders"],
        "poset_reproduced": mot["poset_reproduced"],
    }

    # Third instantiation: curriculum learning (NON-probabilistic, no Fisher info)
    cur = verify_curriculum_instantiation()
    print_("\n[Third instantiation] curriculum learning (no likelihood): "
          "S0 -> {S1, S2} -> S3")
    print_(f"  linear extensions           : "
          f"{[' -> '.join(e) for e in cur['linear_extensions']]}")
    print_(f"  realized acquisition orders : "
          f"{[' -> '.join(o) for o in cur['realized_orders']]}")
    print_(f"  every order is an extension  : "
          f"{cur['all_realized_are_linear_extensions']}")
    print_(f"  both extensions realized     : {cur['both_extensions_realized']}")
    print_(f"  foundational first, comp. last: "
          f"{cur['foundational_first'] and cur['composite_last']}")
    report["curriculum_instantiation"] = {
        "linear_extensions": cur["linear_extensions"],
        "realized_orders": cur["realized_orders"],
        "poset_reproduced": cur["poset_reproduced"],
    }

    # Figures
    print_("\n[7/8] Figures (Fig. 3c sweeps a 26 x 30 phase grid, ~10 s) ...")
    print_("[Figures] writing to", fig_dir)
    figure1_factorization(resA, f"{fig_dir}/Figure1_Factorization.png")
    figure2_accumulation_and_law(resB, accB, f"{fig_dir}/Figure2_RecoveryLaw.png")
    figure3_ordering(grid, dag, f"{fig_dir}/Figure3_Ordering.png")
    figureS1_saturation(f"{fig_dir}/FigureS1_Saturation.png")
    figureS2_counterexample(f"{fig_dir}/FigureS2_Counterexample.png")
    figureS3_motor(mot, f"{fig_dir}/FigureS3_Motor.png")
    figureS4_curriculum(cur, f"{fig_dir}/FigureS4_Curriculum.png")
    print_("  wrote Figure1_Factorization.png, Figure2_RecoveryLaw.png, "
          "Figure3_Ordering.png, FigureS1_Saturation.png, "
          "FigureS2_Counterexample.png, FigureS3_Motor.png, "
          "FigureS4_Curriculum.png")

    ok = (resA["support_reproduces_prerequisite_graph"]
          and resA["max_rel_err_analytic_vs_product"] < 1e-9
          and resA["max_rel_err_montecarlo"] < 0.05
          and accB["matches"]
          and resB["trajectories_coincide"]
          and resB["both_are_saito_loop"]
          and grid["n_admitted"] == 1
          and only[0] == CLINICAL_SAITO_LOOP
          and dag["poset_reproduced"]
          and mot["poset_reproduced"]
          and cur["poset_reproduced"]
          and ce["child_reaches_milestone_first"]           # counterexample reproduced
          and rc["non_increasing"]["graph_order_violations"] == 0  # sufficient condition holds
          and op["onset_order_violations"] == 0             # onset precedence unconditional
          and gf["same_admissible_set"]                     # Proposition A.3
          and gf["min"]["order_violations"] == 0            # Thm 1(b) holds for a non-product gate
          and comp["complete_at_small_ratio"])              # Theorem 2(b) completeness
    print_("\n" + "=" * 74)
    print_("[8/8] done in %.0f s" % (time.time() - _t0))
    print_("OVERALL: all analytical claims of Theorems A and B reproduced =", ok)
    print_("=" * 74)
    report["overall_pass"] = bool(ok)
    with open(f"{fig_dir}/derivation_verification_report.json", "w") as fh:
        json.dump(report, fh, indent=2, default=float)
    return report


def _resolve_out_dir(argv):
    """Pick an output directory from argv while ignoring the flags that Colab /
    Jupyter inject (e.g. ``-f /root/.../kernel-XXXX.json``), which otherwise get
    mistaken for a directory and crash savefig."""
    for arg in argv[1:]:
        if arg.startswith("-"):        # skip flags such as -f
            continue
        if arg.endswith(".json"):      # skip the kernel connection file
            continue
        return arg
    return "."


if __name__ == "__main__":
    main(_resolve_out_dir(sys.argv))
