import numpy as np
from kasner_scattering.propagation.conformal import geometric_spectrum, nonlinear_force
from kasner_scattering.propagation.second_event import diagonal_rate, diagonal_spectrum, observables
from kasner_scattering.propagation.response_experiment import nonlinear_initial

def test_diagonal_geometric_normalization_and_rate():
    q = np.array([-0.1, -0.3, 0.2, 0.0, 2.0, 0.0, 0.0])
    v = np.array([-0.05, -0.9, 0.03, 0.0, 0.15, 0.0, 0.0])
    p = diagonal_spectrum(v)
    geom = geometric_spectrum(q, v, np.zeros(3))
    assert np.allclose(geom, np.r_[np.sort(p[:3]), p[3]], atol=1e-15)
    a = np.array([0.001, 0.03, -0.002, 0.0, -0.004, 0.0, 0.0])
    epsilon = 1e-05
    numeric = (diagonal_spectrum(v + epsilon * a) - diagonal_spectrum(v - epsilon * a)) / (2 * epsilon)
    assert np.max(abs(numeric - diagonal_rate(v, a))) < 2e-11

def test_vacuum_kasner_rate_zero():
    q = np.zeros(7)
    v = np.array([-0.1, -0.9, 0.0, 0.0, np.sqrt(0.0925), 0.0, 0.0])
    a = nonlinear_force(q, v, np.zeros(7))
    assert np.max(abs(diagonal_rate(v, a))) < 1e-15

def test_source_partition_and_common_curve_spectra():
    grid = np.arange(-400, 401) * 0.4
    state, charge = nonlinear_initial(grid, 1e-05)
    obs = observables(state, 0.4, np.array([350, 400, 450]), charge)
    assert np.max(obs['constraint']) < 1e-11
    assert np.max(obs['axis_error']) < 1e-12
    assert np.max(abs(obs['total_rate'] - obs['matter_rate'] - obs['spatial_rate'] - obs['vacuum_rate'])) < 1e-16

def test_balanced_four_dimensional_jets_preserve_normalized_residuals():
    from kasner_scattering.propagation.four_dimensional import homogeneous_jets, residuals
    from kasner_scattering.propagation.second_independent import balanced_residuals
    q = np.array([-0.1, -2.0, 0.2, 0.0, 1.0, 0.01, 0.0])
    v = np.array([-0.05, -0.9, 0.03, 0.0, 0.15, 0.002, 0.0])
    point, first, second = homogeneous_jets(q, v, np.zeros(3), (0.05, 0.0, 0.0))
    second[0, 0, 4] += 0.001
    second[0, 0, 1] += 0.002
    a, b = (residuals(point, first, second), balanced_residuals(point, first, second))
    assert abs(a['einstein_relative'] - b['einstein_relative']) < 1e-12
    assert abs(a['scalar_over_expansion_squared'] - b['scalar_over_expansion_squared']) < 1e-12

def test_magnetic_gradient_belongs_to_local_matter():
    from kasner_scattering.propagation.evolution import constraint_initial_data
    grid = np.arange(-20, 21) * 0.1
    q, v = (np.zeros((7, len(grid))), np.zeros((7, len(grid))))
    q[4] = 1.0
    q[5] = 0.001 * grid
    v[0], v[4] = (-0.1, 0.2)
    state = constraint_initial_data(grid, q, v)
    obs = observables(state, 0.1, np.array([20]), (0.0, 0.0, 0.0))
    magnetic = obs['magnetic'][0]
    velocity = state[7:14, 20]
    force = np.zeros(7)
    force[2] = 2 * magnetic * (velocity[0] + velocity[1]) ** 2
    force[4] = -magnetic * (velocity[0] + velocity[1]) ** 2
    assert np.max(abs(obs['matter_rate'][0] - diagonal_rate(velocity, force))) < 1e-14

def test_plateau_rejects_cancellation_of_large_sources():
    from kasner_scattering.propagation.second_analysis import admissible
    metrics = {'drift': 1e-06, 'max_em_fraction': 1e-06, 'max_spatial_fraction': 1e-06, 'matter_rate_absolute_integral': 0.01, 'spatial_rate_absolute_integral': 0.01, 'max_axis_error': 0.0}
    assert not admissible(metrics, 0.0001)
