# recovery-order-theory

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21351889.svg)](https://doi.org/10.5281/zenodo.21351889)

Numerical verification code for

> **Recovery order in adaptive systems is set by dependency structure**
> Hiroki Saito (2026)

This repository contains the single, self-contained script that **validates** (it does not
prove) the analytical results of the paper, and regenerates every figure in it.

The proofs live in the paper (Methods, Theorems A, B and 1-3, Proposition A.3). The numbers
here confirm that concrete minimal models behave exactly as those theorems require.

---

## Quick start

```
git clone https://github.com/entrance4-png/recovery-order-theory.git
cd recovery-order-theory
pip install -r requirements.txt
python recovery_order_verification.py figures/
```

Runtime: roughly 30-180 s depending on the machine. The script prints a progress marker
(`[1/8]` to `[8/8]`) for each stage, and ends with a single pass/fail line:

```
OVERALL: all analytical claims of Theorems A and B reproduced = True
```

Only `numpy` and `matplotlib` are required. No network access, no data files, no seeds to set:
every result is deterministic.

---

## What is checked

| Claim in the paper | Check |
| --- | --- |
| **Theorem A**. Fisher information factorizes through the dependency graph, `I_k = c_k · Π_{j ∈ anc(k)} r_j` | closed form vs Monte-Carlo (variance of the score) vs the product, over a random grid of availability levels; and the zero-pattern of the Fisher field vs the graph |
| **Theorem B1**. posterior precision accumulates Fisher information, `dP_k/dt = I_k` | exact conjugate precision addition vs the accumulation ODE |
| **Theorem B2/B3**. the recovery law `dr_k/dt = c'_k · A_k · (1 - r_k)` is derived, not assumed | integrating the Fisher-first pipeline through the derived link vs the companion law |
| **Theorem 1(a)**. onset precedence is **unconditional** | 120 random rate vectors, including strongly increasing ones |
| **Theorem 1(b)**. milestone precedence holds when rates do not increase along edges (`c_k ≤ c_j`) | 108 samples with non-increasing rates vs 108 with increasing rates |
| **Sharpness**. the rate condition cannot be dropped | the counterexample (`r0 = 0.5`, `c_j = 1`, `c_k = 100`) and the reversal boundary |
| **Proposition A.3**. the gate enters only through its **support** | the same graph under the product gate and the min gate (a different t-norm with the same support) |
| **Theorem 2(a)**. soundness: every realized order is a linear extension | chain (243-point pre-declared grid) and branch |
| **Theorem 2(b)**. completeness: every linear extension is realized | 30 random DAGs; rates decaying along a target extension σ |

## Figures

`recovery_order_verification.py <outdir>` writes the paper's figures to `<outdir>`:

| File | In the paper |
| --- | --- |
| `Figure1_Factorization.png` | Fig. 1 |
| `Figure2_RecoveryLaw.png` | Fig. 2 |
| `Figure3_Ordering.png` | Fig. 3 |
| `FigureS1_Saturation.png` | Supplementary Fig. 1 |
| `FigureS2_Counterexample.png` | Supplementary Fig. 2 |
| `FigureS3_Motor.png` | Supplementary Fig. 3 |
| `FigureS4_Curriculum.png` | Supplementary Fig. 4 |
| `derivation_verification_report.json` | machine-readable summary |

## Reproducibility

Every random draw goes through `numpy.random.default_rng` with a fixed seed, and the
ancestor sets are sorted, so results are **bit-reproducible across machines and across
`PYTHONHASHSEED` values**. The expected output is:

```
Monte-Carlo == closed-form        : max rel err 2.49%
support(I_k) == prerequisite graph: True
distinct orderings                : 1 of 120
non-increasing rates              : 0/108 violations
increasing rates                  : 102/108 violations
onset precedence                  : 0/120 violations
reversal boundary                 : r0=0.0:1.64 ... r0=0.7:1.22
product gate / min gate           : 0/200, 0/200 violations; same admissible set
completeness                      : 30/30 targets realized
```

## Note on what the code does *not* do

The dependency graph is an **input**, not an output. The ancestor sets are supplied as the
premise of Theorem A; the checks confirm that the Fisher information has the *derived*
support and magnitude. They do **not** discover the edges by search. Identifying the graph
from a system's geometry is the paper's central open problem.

## Citation

If you use this code, please cite the paper and the archived release:

> Saito, H. recovery-order-theory: numerical verification for "Recovery order in adaptive
> systems is set by dependency structure". *Zenodo* <https://doi.org/10.5281/zenodo.21351889> (2026).

The concept DOI `10.5281/zenodo.21351889` always resolves to the latest version. The current
release is v1.0.1 (`10.5281/zenodo.21405028`).

## License

MIT. See `LICENSE`.
