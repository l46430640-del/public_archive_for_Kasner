# Data dictionary

Input protocols are stored once in `data/inputs/`. Metadata for native
predictions and independent evolutions are stored under `data/native/`,
including the metadata for reduced observation subsets under `data/response/`.
The directed `linear.npz` corresponds to `prediction_N512_c0.4`.
Separate domain-size runs retain their own arrays even when the observation
values agree bit for bit; their domain parameters remain distinct.

JSON stores parameters and diagnostics, CSV displayed values, and NPZ lossless numerical arrays. Load NPZ with `allow_pickle=False`. Complex input matrices use a final `[real, imaginary]` axis. Historical transfer rows use objects with `re` and `im` keys.

## Input and geometry

`delta_target` selects a background; `measured_delta` is computed from its charge-to-mass ratio. `canonical` and `canonical_momentum` retain both odd-parity channels in basis `(V_g,V_e)`, harmonic degree two. Polarization maximizes the reference magnetic row at delta=1e-6 with unit Euclidean channel norm. `carrier.json` specifies the radial normalization.

The interfaces remain `PacketSpec`, `MatchingData`, and `EvolutionResult`. A null platform/scattering value means unaccepted or unevaluated. Local constraints, linear horizon readiness and complete nonlinear matching are separate fields. Missing polar data and incompatible transverse flux are recorded.

The evolution state has shape `(17, grid_points)`: seven fields `(r,sigma,P,Q,psi,a_1,a_2)`, their seven velocities, then two orbit connections and the longitudinal Maxwell potential. Here `r=log(rho)` is the orbit area variable. Three conserved charges fix auxiliary momenta. See `conformal.py` for the reconstructed metric.

Spectra contain three sorted geometric spatial exponents followed by the canonical scalar exponent, with `varphi=sqrt(2)*psi`. Magnetic controls retain physical axis order before sorting.

## Response

Reference endpoint files have 175 shared nodes; directed files have 65 curves at three fixed times. `coherent` includes all quadratic parts; `omitted_cross` omits only the mixed diagnostic term. Evolution never filters frequencies.

`quadratic_parts` is ordered carrier squared, mixed, soft squared and includes `N^2`, `lambda`, and `lambda^2`. Its component axis is zero for reference endpoints and one for directed three-time files. Both error fractions use denominator `max_y abs(coherent)`. `p0` normalizes displayed responses. Reference endpoints use the same-model final zero-packet scalar value; directed comparisons interpolate that background at each fixed time.

`linear.npz` retains propagated quadratic coefficients and background. Native `states` contains two linear fields and their momenta, followed by three sets of eight quadratic geometric/scalar variables. Expansion breakdown and numerical discretization error have separate diagnostics.

## Exchange

Each history retains all saved times on five observation curves. `magnetic` and `electric` are normalized transverse fractions; `volume` is logarithmic volume time. Rate arrays split the geometric spectral evolution into matter, spatial and vacuum contributions. `windows.json` fixes peak and half-height endpoints, integration windows and platform screening.

The five columns of each curve's `rates` array are electric forcing, magnetic forcing, remaining matter, spatial terms and vacuum geometry. `weights @ rates` reconstructs the impulses with their saved trapezoidal measure. Finite quadrature closure is retained.

## Magnetic controls

Files named `d*_phi*_{DOP853,Radau}_r*` store prescribed inputs, peak roots, coupling slopes, platform windows and errors. DOP853 uses relative tolerances 1e-9, 1e-11 and 1e-13; Radau uses exact rescaled variables at 1e-11. All eight acceptance decisions and window/source checks appear in `verification.json`.

Historical frequency and wall-slope meanings are documented in [provenance](provenance.md). Embedded historical hashes retain their original meaning; current file hashes appear in the manifest.
