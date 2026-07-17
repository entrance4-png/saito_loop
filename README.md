# saito_loop

Verification code for

> **The Saito Loop, a computational theory of hierarchical recovery in catatonia**
> Hiroki Saito (2026)

Preprint: `<medRxiv DOI>` · Archive: [10.5281/zenodo.21387507](https://doi.org/10.5281/zenodo.21387507) (v2.0.2, the version used in the paper)

This repository contains one self-contained script that **checks** (it does not prove)
the analytical results of the paper and regenerates its figures.

The proofs live in the paper (Methods, Theorems 1–3). The code confirms that a concrete
minimal model behaves exactly as those theorems require.

---

## Quick start

```bash
git clone https://github.com/entrance4-png/saito_loop.git
cd saito_loop
pip install -r requirements.txt
python saito_loop.py            # report + figures in the current directory
python saito_loop.py ./out      # figures written to ./out
```

Python >= 3.9. Only `numpy`, `networkx` and `matplotlib` are required. No network access
and no data files: every result is deterministic.

The run ends with a single pass/fail line:

```
OVERALL: Part A deductions = True; Part B sole passer = Saito = True; Part C bounded partitions verified = True
```

---

## What is checked

The script has three parts.

**Part A — two-stage deduction, robustness, figures**

| Claim in the paper | Check |
| --- | --- |
| **Theorem 2**. the prerequisite edges are derived, not assumed | the edges are reconstructed from the joint Poisson likelihood by the mean-Jacobian zero-column test, without the hand-written edge list |
| the chain is the only topological order | `distinct orderings : 1 of 120` |
| **Theorem 1**. exactly five roles are milestone-identifiable | every pair-merge loses a separation; refinement beyond five is unidentifiable |
| the milestone order follows under uniform onset | no crossings in the recovery dynamics |
| robustness | a pre-declared 729-point grid over nuisance parameters |
| **Theorem 3**. non-uniform onset permits only a restricted subset | `admitted orderings : 8 of 120`; πv_slow never first |
| the rate condition is necessary | reversal outside the gating-dominated regime is confirmed, with the boundary located |

**Part B — dynamical audit of four (P,S) families** against three prespecified signatures
(canonical order, dissociation, directionality). Only `saito_prerequisite` passes all three.

**Part C — bounded classification**: the one-edit neighbourhood N₁(M) and the Boolean
(P,S) space, both partitioned exhaustively. `universal_uniqueness_claimed: False`.

---

## Expected output

```
[Stage 0] derived edges     : pi_s->beta, beta->pi_m, pi_m->pi_vfast, pi_vfast->pi_vslow
          DECISIVE beta=0  -> I_joint(pi_m)=0 (non-identifiable; no leak)
[Robustness] 729 points, distinct orderings : 1 of 120
[Competing]  independent-rate: 93 distinct orderings
             common-severity : spread = 0.000 (no dissociation)
             Saito Loop      : spread = 3.540 (dissociations)
[Non-uniform onset] admitted orderings : 8 of 120; pi_vslow never first : True

family,grid,canonical,dissociation,directional,all3,distinct_orders
independent_rate,729,7,729,0,0,120
common_severity,729,0,0,0,0,1
saito_prerequisite,729,729,729,729,729,1
hybrid,729,729,729,0,0,1
```

Figures written: `Figure1_DAG.png` (Fig. 1), `Figure2_Recovery.png` (Fig. 2),
`Figure3_Onset.png` (Fig. 3), `Figure4_Competing.png` (Fig. 4), plus
`verification_report.json` (machine-readable summary).

---

## What the code does *not* do

The prerequisite edges are reconstructed from the likelihood, but the *likelihood
architecture itself* instantiates the modelling commitments M1–M3. Those functional forms
are commitments, not mathematical necessities of active inference. Uniqueness of the graph
is conditional on the class 𝔐\* of observation models sharing the same structural zero set.

The code does not discover the graph by search, and it is a consistency check, not a proof.
The deductive conclusions are independent of simulation length, random seed, grid resolution
and context number.

Uniqueness of the *temporal* order additionally requires the gating-dominated regime
(no downstream rate exceeds its prerequisite's). Outside it, a descendant can reach
threshold first; the script locates that boundary. A graph-general characterization
remains future work. **Universal uniqueness is not claimed.**

---

## Citation

If you use this code, please cite the paper and the archived release. See `CITATION.cff`.

## License

MIT. See `LICENSE`.
