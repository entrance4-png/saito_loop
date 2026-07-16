# Changelog

All notable changes to `saito_loop.py` are documented here.

## v2.0.1 — Documentation alignment (behaviour identical to v2.0.0)

No change to computation, logic, function names, or results; the
machine-readable `verification_report.json` is byte-identical to v2.0.0.

### Changed
- Relabelled the `theorem1_irreducibility` routine to match the manuscript's
  final Theorem 1: the **count of five is derived structurally** from the
  free-energy factorization (Methods, Lemma 1), and this routine only verifies
  the **milestone-side selection** among theory-admissible cardinalities
  (a coarser decomposition collapses an observed dissociation; a finer one adds
  a precision with no distinct milestone). The console line now reads
  `=> milestone-side selection of five`. This removes wording in the previous
  docstring that described the check as a formal minimality "proof".

## v2.0.0 — Corrected analysis (supersedes v1.0.0)

**v1.0.0 is superseded and should not be used to reproduce the published results.**

### Fixed
- **Cross-channel information leakage (critical).** Identifiability is judged
  from the **joint** Fisher information of all observation channels
  (`joint_fisher`, mean-Jacobian zero-column test), not from each parameter's
  own-channel diagonal. The observation model is made *motivationally separable*
  so that `β = 0` genuinely leaves `π_m` non-identifiable in the joint likelihood.
- **Prerequisite edges are derived, not assumed** (`derive_edges_from_generative_model`)
  from an explicit **Poisson** likelihood; the hand-written list is only an audit
  target (`EXPECTED_EDGES_FOR_AUDIT`).
- **Correct direct edge `π_m → π_v_fast`** (previously only `β → π_v_fast`).
- **Fisher information derived from the score**, with a Monte-Carlo check.
- **Recovery order is conditional on rates**: unconditional "any positive rates"
  removed; gating-dominated condition (`c_child ≤ c_parent`), reversal boundary
  (~1.64 at q = 0.9) and a two-node counterexample added.
- **Theorem 1** backed by a milestone-side selection check
  (`theorem1_irreducibility`).
- **JSON serialization bug** fixed (`_to_jsonable`).

### Changed
- Verdict reports separate, regime-scoped checks instead of one aggregate pass.

## v1.0.0 — Initial release (deprecated)
Superseded by v2.0.0.
