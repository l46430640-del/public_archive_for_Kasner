# Coherent Selection of Critical Kasner Response

Data and numerical source for the local Einstein-Maxwell-scalar mechanism in *Coherent Selection of Critical Kasner Response*, version 2.0.0.

Reconstruction coverage and comparison criteria are listed in
[validation](docs/validation.md).

The central comparison fixes the initial reduced electromagnetic energy. A weak coherent component changes the first scalar response, and two fixed inputs reverse their response ordering between two critical backgrounds. This archive supplies fields, momenta, quadratic predictions, nonlinear observations, numerical controls, first exchange histories, and eight homogeneous magnetic controls.

## Reproduce

Use Python 3.11 in a fresh environment from this checkout:

```sh
python -m venv .venv
```

Activate the environment for your shell, then:

```sh
python -m pip install -e ".[test]"
python -m kasner_scattering verify
python -m pytest
python -m kasner_scattering plot --output build/figure
```

Verification checks all frozen file hashes and recomputes displayed comparisons from arrays. Tests check equations, constraints, normalization, interference, discretization and Fourier transforms.

```sh
python -m kasner_scattering rebuild --study directed --output build/directed
python -m kasner_scattering rebuild --study reference --output build/reference
python -m kasner_scattering rebuild --study exchange --output build/exchange
python -m kasner_scattering rebuild --study wall --output build/wall
python -m kasner_scattering rebuild --study carrier --output build/carrier
python -m kasner_scattering rebuild --study restricted-wall --output build/restricted
```

Use `--study all` for the fixed complete list. Each output must be new and strictly below `build/`. The commands reject overwriting frozen evidence. Integration completion and successful comparison are distinct states. See [validation](docs/validation.md) for measured coverage and costs.

Reconstruction uses one BLAS thread and verifies the active setting, including
when NumPy was imported earlier. Run each study in the main thread of a
dedicated Python process. The thread setting is part of the numerical recipe;
changing native-library operation ordering can affect derivative residuals
near the floating-point noise level.

## Evidence

| Location | Purpose |
|---|---|
| `data/inputs/` | Complex carrier data, background points, prescribed inputs and windows |
| `data/response/` | Common-node comparisons, convergence evidence and exchange integrals |
| `data/native/` | Propagated fields and momenta, continuous-frequency checks and histories |
| `data/wall/` | Eight single-magnetic-axis controls and their independent integrations |
| `data/tables/` | Displayed values and complete spatial profiles |
| `data/trajectories.jsonl` | 495 historical restricted Hamiltonian records |
| `src/kasner_scattering/` | Equations, solvers, reconstruction and verification |
| `figures/response_profiles.*` | The current two-panel response figure |

See [methods](docs/methods.md), [data dictionary](docs/data_dictionary.md), [evidence map](docs/evidence.md), and [provenance](docs/provenance.md).

Propagation concerns finite-time dynamics with two Killing fields. Linear horizon data and locally constrained initial data have separate validity states. Full nonlinear horizon matching and outgoing packet platforms remain unestablished. Homogeneous controls supply a conditional reflection reference.

Code: [BSD 3-Clause](LICENSE). Data, figures and documentation: [CC BY 4.0](LICENSE-DATA).
