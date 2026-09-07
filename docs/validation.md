# Validation

All six prescribed reconstruction groups pass their frozen-observation
comparisons. [Machine-readable results](validation.json) give the individual
inputs, coverage, diagnostic values and comparison decisions.

| Study | Completed comparisons |
|---|---|
| Reference response | 27 prescribed jobs, including phase, domain, time-step, fixed-Kasner and continuous-frequency controls |
| Response on the second background | 24 prescribed jobs, including four independent evolutions and two continuous-frequency resolutions |
| First exchange | 19 jobs: fifteen complete histories and four independent interaction segments |
| Single magnetic axis | Eight controls, each at three DOP853 tolerances and with rescaled Radau, for 32 integrations |
| Linear carrier | Both backgrounds and 22 regular-horizon transfer evaluations |
| Restricted Hamiltonian | All 495 records; the reconstructed normalized outgoing-spectrum difference is zero |

## Numerical Criteria

All reconstruction limits in `comparison.py` are unchanged. Array agreement
tests implementation reproducibility; the supplied spatial, temporal and
independent-discretization comparisons estimate solution error.

On the second background, the fine independent calculations satisfy the
reported ceilings: relative Einstein residual 9.21e-10, relative Maxwell
residual 1.43e-10, scalar residual divided by squared expansion 8.86e-13,
absolute reduced-energy residual 9.35e-12, and independent spectrum difference
1.29e-10. The fine long-exchange spectrum differences remain below 7.5e-9.
Every previously stored residual field in the ten independent result files is
reproduced exactly. The reference reconstructions also return four energy
and work fields absent from the earlier records; these are additional outputs.

One active BLAS thread is part of the reconstruction recipe. Tests check
that the setting is enforced even after NumPy initialization. Derivative-fit
residuals close to the arithmetic floor remain distinct from estimates of
the physical solution error.

## Input Independence

Numerical runs used an isolated copy of the public source and data with a
fresh dependency environment. Reads from the source repositories were
blocked, as were network and subprocess access during these calculations.
This check covers
all directed, magnetic-control, carrier and restricted-Hamiltonian jobs.
For the reference study it covers the fine prediction, both decisive fine
nonlinear inputs and the fine independent segment with a regenerated
checkpoint. For exchange it covers the domain and constant-coupling controls
and all four independent segments. The other prescribed histories were
already reconstructed and compared. Per-job flags in `validation.json`
record the coverage.

This file-access check is separate from independent numerical discretization.
The latter has the initial-surface and checkpoint scope specified in the
[evidence map](evidence.md). Reconstructed derivative neighborhoods are new
calculations, while the archive preserves the available historical residual
records and arrays.

## Frozen Evidence And Figure

Frozen verification recovers all displayed comparisons, 240 profile nodes,
fifteen exchange curves, 981 finite arrays, the regular-horizon checks and
eight accepted magnetic controls. The 42 focused tests cover equations,
constraints, normalization, interference, transforms, validity states,
immutable inputs and active BLAS control. Deliberate data changes are detected
by both the manifest and observable checks.

The two-panel figure is regenerated from all 175 and 65 profile nodes.
Its PDF and PNG reproduce the archived bytes. Visual inspection and
PyMuPDF text-box checks find no intersecting labels or out-of-page text.
The hashes in `validation.json` bind these checks to the figure assets.

Frozen verification takes seconds. Complete propagation and independent
controls require tens of minutes or longer per study, depending on the
numerical environment. `rebuild --study all` runs the fixed complete list;
each study can also be run separately into a new directory below `build/`.
