# Vanishing-energy first-wall Kasner scattering

This repository is a self-contained reproduction package for the
scientific results underlying *Vanishing-Energy Taub-Rescaled Kasner
Scattering inside Scalarized Black Holes*. It contains the equations,
numerical implementation, frozen machine-readable results, and compact
reproduction figures. 

The package follows one causal chain: regular odd-parity data on the future
horizon excite a soft magnetic component; near the scalarization threshold
its horizon amplitude and flux energy vanish under a double scaling; the
large-field coupling produces an essential first-wall clock; and the
contracting magnetic layer retains a finite tangent-space Kasner reflection.

For any question, contact: liyikun@xao.ac.cn

## Install

Python 3.11 is required. From a fresh virtual environment:

```console
python -m pip install -e ".[test]"
```

## Reproduce

Verify the committed results and analytic identities:

```console
python -m kasner_scattering verify
```

Recompute the threshold, eleven backgrounds, 121 odd-transfer records, 495 first-wall
trajectories, finite-wall map, and coupling controls into an ignored output
directory:

```console
python -m kasner_scattering rebuild --output build/full
```

Regenerate the public summary figure:

```console
python -m kasner_scattering plot --output build/figures
```

An optional Wolfram Language calculation independently checks the exact
threshold, three transfer frequencies, and the analytic inner reflection:

```console
python -m kasner_scattering crosscheck-wolfram
```

On Windows, `reproduce.ps1` provides the same workflow with `-Full` and
`-Wolfram` switches.

## Contents

- `data/` contains five normalized scientific artifacts and their hashes.
- `src/kasner_scattering/` contains the compact numerical implementation.
- `figures/results_summary.*` is generated entirely from committed data.
- `docs/methods.md` records equations and normalizations.
- `docs/data_dictionary.md` defines public fields and comparison tolerances.

The result concerns the first isolated Taub-degenerate magnetic transition.
A broader Einstein-Maxwell-scalar evolution belongs to the same local limit
when its additional generalized-Kasner shift is smaller than
`sqrt(delta)`.

## Licensing

Source code is distributed under the BSD 3-Clause License. Data, figures, and
documentation are distributed under CC BY 4.0; see `LICENSE-DATA`.
