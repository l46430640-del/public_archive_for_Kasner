# Claim and evidence map

| Comparison | Frozen evidence | Reproduction |
|---|---|---|
| Equal-energy response and ordering | `response/reference/{weak,near,strong}.npz`, `response/directed/{weak,strong}.npz`; input energy coefficients and protocols | `reference`, `directed`; `verification.response_checks` |
| Full spatial profiles and fixed-time table | `tables/response_profiles.csv`, `tables/letter_ordering.csv` | `plot`; `verification.table_checks` |
| Mixed signed source | `tables/signed_sources.json`, quadratic coefficient arrays | `verification.response_checks`, `table_checks` |
| Space, time and domain errors | Three grid levels and independent time steps in each response group; long histories under `native/exchange` | `verification.convergence_checks`; prescribed study lists |
| Phase and fixed-Kasner controls | Reference phase pi/2 arrays and fixed-Kasner quadratic prediction under `native/reference` | `reference`; phase and normalization tests |
| Continuous band and transform precision | Two continuous-mode resolutions for each background; `compact_spectrum.py` and its high-precision tests | `reference`, `directed`; Fourier tests |
| First magnetic exchange | Fifteen curve files, three complete histories, fixed windows and weights | `exchange`; `verification.exchange_checks` |
| Geometric constraints and direct four-dimensional residuals | Independent comparison records and available native fields | `reference`, `directed`, `exchange`; `four_dimensional.py` |
| Conditional homogeneous reflection | Eight controls at four integration settings, actual peak roots and window checks | `wall`; `wall_verification.py` |
| Carrier normalization and physical units | `inputs/carrier.json`, `matching_data.json`, `regular_horizon.json`, background sequence | `carrier`; `verification.carrier_checks` |
| Restricted Hamiltonian comparison | 495 records, historical backgrounds, and the radial transfer source | `restricted-wall` |

Paths in the evidence column are relative to `data/`. Source paths are relative to `src/kasner_scattering/` or its `propagation/` subpackage.

## Independent coverage

Directed FD6/DOP853 comparisons evolve from the same constrained initial surface. Reference and long-exchange independent calculations share an evolved checkpoint and independently discretize the subsequent interaction segment. They are not full-history independent replications. Long exchange also has complete FD4/RK4 histories at three grids.

Historical directed and long-segment checks retained residual values and spectral observations but did not retain their full time-derivative neighborhoods. Their source reconstructs new neighborhoods on rerun. These new calculations must not be described as recovered historical arrays. Reference independent time slices are retained and permit direct neighborhood reconstruction.

Four-dimensional residuals concern the reconstructed two-Killing field sector. They do not test omitted angular dependence or complete nonlinear horizon matching. Linear carrier data retain all stored odd components; only compatible components enter the local constrained packet experiment.

The propagated packet has no accepted outgoing platform or scattering error. The isolated magnetic controls compare two platforms on the same trajectory, using the actual peak slope and the stated isolation conditions.
