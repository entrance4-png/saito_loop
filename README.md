# saito_loop

Code accompanying **"The Saito Loop: A Deductive Theory of Hierarchical Recovery in Catatonia"** (H. Saito).

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Preprint: <medRxiv DOI をここに>

## What it does

Running the package regenerates, from scratch:

- the prerequisite dependency graph and its unique topological ordering (Theorem 2)
- the recovery trajectories and milestone crossings (Fig. 2)
- the 729-point robustness grid (Fig. 4c)
- the competing-model comparison (independent-rate, common-severity, hybrid; Fig. 4a,b)
- the non-uniform-onset sweep (Theorem 3; Fig. 3)
- all figures in the manuscript and a machine-readable verification report

## Install

```bash
git clone https://github.com/<user>/saito_loop.git
cd saito_loop
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

Python >= 3.10. Dependencies are pinned in `requirements.txt`.

## Reproduce everything

```bash
python -m saito_loop.run_all --out results/
```

This writes all figures to `results/figures/` and `results/verification_report.json`,
in which every declared deductive claim is reported as passed or failed.
Random seeds are fixed; the run is deterministic.

Expected runtime: < X minutes on a laptop.

## Modules

| module | purpose |
|---|---|
| `model` | generative model and the five precisions |
| `identifiability` | likelihood-flattening / structural identifiability test |
| `topology` | dependency graph and topological ordering |
| `recovery` | information-limited recovery law, milestones |
| `theorem` | machine-checked verification of Theorems 1–3 |
| `competing_models` | independent-rate, common-severity, hybrid families |
| `figures` | figure generation |

## Citation

If you use this code, please cite the software (Zenodo DOI above) and the paper.
See `CITATION.cff`.

## License

MIT (see `LICENSE`).
