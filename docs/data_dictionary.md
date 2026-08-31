# Data dictionary

All JSON files use sorted keys and LF line endings. Every scientific record
contains `record_sha256`, calculated from the record before that field is
inserted. Container files contain `content_sha256` calculated in the same way.

## `backgrounds.json`

- `threshold`: exact Legendre root, direct compactified integrations, and
  finite-cutoff convergence values.
- `high_precision_comparisons`: independent high-precision comparisons at
  `delta=10^-8` and `10^-9`.
- `records`: eleven nonlinear backgrounds on the half-decade grid. `target_delta` is the requested
  branch position and `measured_delta` is reconstructed from `q/q_c-1`.
- `sigma_k`: geometric interior matching coordinate.
- `beta_k`, `phi_k`, `chi_k`, `h_k`: background fields at that surface.

## `transfers.json`

- `records`: 121 soft magnetic rows at eleven frequencies on eleven critical
  backgrounds. Complex values use `{re, im}`.
- `frequency_sensitivity_row`: derivative in a parallel-transported common
  horizon phase.
- `strong_equation_defect`: nonzero centered-difference defect of the
  integrated radial solution.
- `neighborhoods`: explicit frequency and `CP^1` lower bounds.
- `independent_check`: separately integrated high-precision transfer rows.

## `trajectories.jsonl`

Each of 495 lines is one complete physical input and result:

- `input`: `delta`, carrier frequency, `kappa`, polarization, horizon
  amplitude, local magnetic amplitude, and wall fraction.
- `packet_energy_tau_1`: Gaussian flux energy at `tau=1`.
- `clock`: physical peak time, essential normalization, prediction, and error.
- `plateaus`: incoming and outgoing generalized Kasner exponents, plateau
  diagnostics, Hamiltonian constraint, and numerical energy drift.
- `finite_q_map`: canonical momentum, Jacobian, predicted output, limiting
  reflection, and comparison with the direct Hamiltonian solution.
- `inner_limit`: rescaled distance from the explicit scalar reflection.

## `controls.json`

`map_value_status` separates a measured two-plateau transition from an upper
bound. `NO_ONSET` and `NO_POST_PLATEAU` never contain a fabricated zero map.

## `summary.json`

This is derived from the four detailed datasets. It collects counts and global
extrema used by the quick verifier and public figure.

## Numerical acceptance

- relative target-`delta` error below `10^-6`;
- odd symplectic-flux residual below `10^-9`;
- positive transfer lower bounds on the stated open set;
- Hamiltonian/Kasner constraint below `10^-10`;
- normalized peak-clock drift below 5 percent;
- direct versus finite-wall map difference below 2 percent;
- Python/Wolfram transfer difference below `10^-8` after common-phase alignment.
