# Provenance and conventions

Numerical arrays retain their original precision and comparison sampling. Selection removes duplicate material and fields outside the scientific purpose of this archive. The manifest assigns every public file a scientific role.

The reference coefficients and directed comparison times were selected from linear predictions before their nonlinear comparisons. Supplied protocols preserve those choices. This describes the research procedure, not public preregistration.

## Historical records

The 495 trajectories retain their original numbers. Their diagonal Hamiltonian evolution retains a perpendicular electric term; the reduced comparison omits it. A general multiaxis electromagnetic configuration in a diagonal homogeneous metric does not satisfy the omitted off-diagonal equations. These records describe a restricted Hamiltonian comparison. The eight single-axis, zero-electric-field controls supply the self-consistent homogeneous reference.

Historical `omega_M` was the coordinate frequency passed directly to the transfer integrator. The current carrier uses `omega=0.2/M`, with horizon radius one. Historical `q_B` was a free-flight proxy. Current magnetic controls locate the actual peak by a root of its derivative and evaluate the coupling slope there. Historical values are not silently reinterpreted.

The 495-row CSV and historical JSON retain their original evaluations of the relative finite-wall discrepancy. Those evaluations differ by at most 3.6e-15; their input delta, clock and constraint values coincide exactly. Verification treats this arithmetic difference separately from integration error.

Earlier archive states remain in Git history; their interpretation is superseded by the current definitions and scope.

## Figure precedent

Dense spatial predictions and nonlinear observations follow the matching data semantics in Dean et al., *PRL* **127**, 135502 (2021): [plotting source](https://github.com/j-m-dean/data_overscreening_and_underscreening/blob/163c9a9adf1843d77230117152e0900db113b0ef/kmc_paper_data/overscreening_figure.py), lines 172-195. Borrowed presentation elements are simulation markers, continuous competing profiles and aligned panels. No third-party code is copied. All 175 and 65 nodes remain in the CSV; only marker density is reduced for display.

## Context

The stationary critical EMS family is developed by [Li, Sun and Yang](https://arxiv.org/abs/2512.19377). Local singularity dynamics and spatially exceptional behavior are studied by [Garfinkle](https://doi.org/10.1103/PhysRevLett.93.161101) and [Andersson et al.](https://doi.org/10.1103/PhysRevLett.94.051101). Electromagnetic amplification is treated by [Sobol et al.](https://doi.org/10.1103/PhysRevD.98.063534); inhomogeneous bounces in another symmetry class by [Warren Li](https://arxiv.org/abs/2408.12434). This archive tests phase-dependent finite-time response along a specified critical family.
