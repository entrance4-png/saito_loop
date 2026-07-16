# Changelog

All notable changes to `saito_loop.py` are documented here.

## v2.0.0 — Corrected analysis (supersedes v1.0.0)

This release corrects and strengthens the identifiability analysis. **v1.0.0 is
superseded and should not be used to reproduce the published results.**

### Fixed
- **Cross-channel information leakage (critical).** Identifiability is now judged
  from the **joint** Fisher information of all observation channels
  (`joint_fisher`, mean-Jacobian zero-column test), not from each parameter's
  own-channel diagonal. The observation model is made *motivationally separable*
  (policy enactment depends on sensory and policy precision only), so that
  `β = 0` genuinely leaves `π_m` non-identifiable in the joint likelihood
  (`∂μ_β/∂π_m = 0`).
- **Prerequisite edges are derived, not assumed.** `derive_edges_from_generative_model`
  recovers `π_s → β → π_m → π_v_fast → π_v_slow` from an explicit **Poisson**
  likelihood; the hand-written edge list is only an audit target
  (`EXPECTED_EDGES_FOR_AUDIT`).
- **Correct direct edge `π_m → π_v_fast`** (previously only `β → π_v_fast` was
  established, which does not linearise the DAG into a chain).
- **Fisher information is derived from the score** of an explicit likelihood,
  with a Monte-Carlo check (`poisson_fisher_monte_carlo`).
- **Recovery order is conditional on rates.** The unconditional "any positive
  rates" claim is removed; a sufficient gating-dominated condition
  (`c_child ≤ c_parent`), the reversal boundary (~1.64 at q = 0.9) and a two-node
  counterexample are added.
- **Theorem 1 minimality** is now a formal, two-directional check
  (`theorem1_irreducibility`): five distinct milestone latencies, every
  pair-merge loses a separation, and a refinement beyond five is unidentifiable.
- **JSON serialization bug** fixed (`_to_jsonable`): the verification report now
  writes cleanly (NumPy `bool_`/scalars coerced).

### Changed
- Verdict now reports separate, regime-scoped checks (Theorem 1 minimality;
  edges derived from the joint likelihood; order in the gating-dominated regime;
  reversal outside it; non-uniform-onset predictions within the regime) instead
  of a single aggregate pass.

## v1.0.0 — Initial release (deprecated)
Initial version accompanying the first manuscript draft. Superseded by v2.0.0.
