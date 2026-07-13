# saito_loop

Code accompanying **"The Saito Loop: A Deductive Theory of Hierarchical Recovery in Catatonia"** (H. Saito).

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Preprint: (medRxiv DOI to be added)

## What it does

`saito_loop.py` is a single self-contained script. Running it reproduces every
result and every figure in the paper, and writes a machine-readable verification
report.

**Part A — the two-stage deduction**

- Stage 1: zeroing-based identifiability → prerequisite DAG → unique topological
  order (Theorem 2). The clinically observed recovery order is *held out* and
  never supplied to this stage.
- Stage 2: information-limited recovery law → comparison theorem → unique
  milestone order under uniform onset.
- A pre-declared 729-point robustness grid (six nuisance hyperparameters × three
  levels).
- Non-uniform-onset sweep: which of the 120 orderings are permitted (Theorem 3).
- Competing models: independent-rate and common-severity.
- Figures 1–4.

**Part B** — reference dynamical audit of the four (P, S) coupling families
against the three prespecified signatures.

**Part C** — formal bounded classification: the one-edit neighbourhood N₁(M) and
the Boolean (P, S) partition.

## Requirements

Python ≥ 3.9 and three libraries:

```
numpy
networkx
matplotlib
```

Install them with:

```
pip install -r requirements.txt
```

## How to run

```
python saito_loop.py            # report printed to the terminal; figures written to the current folder
python saito_loop.py ./out      # figures and report written to ./out
```

Outputs:

| file | contents |
|---|---|
| `Figure1_DAG.png` | Fig. 1 — zeroing dependency matrix and the prerequisite DAG |
| `Figure2_Recovery.png` | Fig. 2 — recovery trajectories and milestones |
| `Figure3_Onset.png` | Fig. 3 — uniform vs non-uniform onset |
| `Figure4_Competing.png` | Fig. 4 — two-tier model-space audit |
| `verification_report.json` | every declared deductive claim, reported as passed or failed |

The run is deterministic (random seeds are fixed inside the script). The final
line printed is:

```
OVERALL: Part A deductions = True; Part B sole passer = Saito = True; Part C bounded partitions verified = True
```

Expected runtime: a few minutes on a laptop.

## What the numerical part does and does not establish

The script is a **consistency check on the analytical derivation, not the
derivation itself.** The prerequisite edges are supplied to the code as the
support of the generative gating; Stage 1 confirms that this encoded structure
reproduces the same dependency graph and the same unique topological order that
Theorem 2 derives analytically. What forces the edges is the likelihood-flattening
argument of Theorem 2, not the simulation. The deductive conclusions are therefore
independent of simulation length, random seed, grid resolution and context number.

Uniqueness is **conditional**: within the one-edit neighbourhood N₁(M) and within
the two-mechanism class C. It is not a claim of universal uniqueness.

## Citation

If you use this code, please cite both the software (Zenodo DOI above) and the
paper. See `CITATION.cff`.

## License

MIT (see `LICENSE`).
