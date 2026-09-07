import numpy as np
import pytest
import sympy as sp
from kasner_scattering.propagation.conformal import auxiliary_velocity, geometric_spectrum, homogeneous_constraint_velocity, nonlinear_force, null_constraints, symbolic_system
from kasner_scattering.propagation.evolution import constraint_initial_data, derivatives, evolve, rk4_step
from kasner_scattering.propagation.four_dimensional import homogeneous_jets, residuals

@pytest.mark.parametrize('coupling', ['exp_square', 'constant'])
def test_full_four_dimensional_homogeneous_equations(coupling):
    x = np.array([0.1, -0.2, 0.1, 0.03, 0.4, 0.01, -0.02])
    charges = (0.03, 0.02, 0.01)
    v = homogeneous_constraint_velocity(x, np.array([-0.04, 0.0, 0.01, -0.02, 0.18, 0.02, -0.01]), charges, coupling)
    result = residuals(*homogeneous_jets(x, v, np.array([0.02, -0.01, 0.03]), charges, coupling), coupling)
    assert result['einstein_relative'] < 1e-12
    assert abs(result['scalar']) < 1e-12
    assert np.max(np.abs(result['maxwell'])) < 1e-12

def test_full_four_dimensional_inhomogeneous_equations():
    x = np.array([0.1, -0.2, 0.1, 0.03, 0.4, 0.01, -0.02])
    v = np.array([-0.08, 0.0, 0.01, -0.02, 0.18, 0.02, -0.01])
    w = np.array([0.0, 0.0, 0.02, 0.01, -0.04, 0.013, -0.017])
    yy = np.array([0.0, 0.01, 0.002, -0.003, 0.004, -0.003, 0.001])
    ty = np.array([0.0, 0.003, -0.002, 0.001, -0.004, 0.003, -0.001])
    charges = (0.03, 0.02, 0.01)
    system = symbolic_system()
    sinv = np.asarray(system['shape'].inv().subs(dict(zip(system['fields'], x))), float)
    flux = v[2] * w[2] + np.exp(2 * x[2]) * v[3] * w[3] + 4 * v[4] * w[4]
    flux += 4 * np.exp(x[4] ** 2 - x[0]) * (v[5:7] @ sinv @ w[5:7])
    w[1] = flux / (2 * v[0])
    acc = yy + nonlinear_force(x, v, w, charges)
    v[1] = null_constraints(x, v, w, acc, ty, yy).mean() / (2 * v[0])
    acc = yy + nonlinear_force(x, v, w, charges)
    assert np.max(np.abs(null_constraints(x, v, w, acc, ty, yy))) < 1e-15
    jac = np.asarray(system['auxiliary'].jacobian(system['fields']).subs(dict(zip((*system['fields'], *system['charges']), (*x, *charges)))), float)
    first, second = (np.zeros((4, 10)), np.zeros((4, 4, 10)))
    first[0] = np.r_[v, auxiliary_velocity(x, charges)]
    first[1] = np.r_[w, [0.02, -0.01, 0.03]]
    second[0, 0] = np.r_[acc, jac @ v]
    second[0, 1] = second[1, 0] = np.r_[ty, jac @ w]
    second[1, 1] = np.r_[yy, [0.003, -0.002, 0.001]]
    result = residuals(np.r_[x, [0.02, -0.01, 0.03]], first, second)
    assert result['einstein_relative'] < 1e-12
    assert abs(result['scalar']) < 1e-12
    assert np.max(np.abs(result['maxwell'])) < 1e-12
    point = np.r_[x, [0.02, -0.01, 0.03]]
    point[7:] = 0.0
    first[1, 7:] = 0.0
    second[1, 1, 7:] = 0.0
    transformed = residuals(point, first, second)
    assert transformed['einstein_relative'] < 1e-12
    assert abs(transformed['scalar']) < 1e-12
    assert np.max(np.abs(transformed['maxwell'])) < 1e-12

def test_scalar_normalization_and_time_orientation():
    ps = 0.03
    scalar_v = np.sqrt(2 * ps - 3 * ps ** 2)
    x = np.zeros(7)
    v = np.array([-2 * ps, -(1 - 2 * ps), 0.0, 0.0, scalar_v, 0.0, 0.0])
    spectrum = geometric_spectrum(x, v, np.zeros(3))
    np.testing.assert_allclose(spectrum, [ps, ps, 1 - 2 * ps, np.sqrt(2) * scalar_v], atol=1e-15)
    assert abs(np.sum(spectrum ** 2) - 1) < 1e-15

def test_fourth_order_derivatives():
    errors = []
    for n in (81, 161, 321):
        y = np.linspace(-2, 2, n)
        d, dd = derivatives(np.sin(y), y[1] - y[0])
        errors.append(max(np.max(abs(d[2:-2] - np.cos(y[2:-2]))), np.max(abs(dd[2:-2] + np.sin(y[2:-2])))))
    assert errors[0] / errors[1] > 15
    assert errors[1] / errors[2] > 15

def test_no_periodic_wrap_and_causal_domain_requirement():
    x = np.zeros(30)
    x[-3] = 1
    d, dd = derivatives(x, 0.1)
    assert not np.any(d[:10]) and (not np.any(dd[:10]))
    with pytest.raises(ValueError, match='causal'):
        evolve(np.linspace(-1, 1, 21), np.zeros((17, 21)), 1.0, 0.01, 0.1)

@pytest.mark.parametrize('coupling', ['exp_square', 'constant'])
def test_accelerated_step_matches_numpy(coupling):
    from kasner_scattering.propagation.accelerated import compiled_step
    y = np.linspace(-8, 8, 81)
    x = np.zeros((7, len(y)))
    x[4], x[5] = (0.3, 0.002 * np.exp(-y ** 2))
    v = np.zeros_like(x)
    v[0], v[4] = (-0.1, 0.2)
    state = constraint_initial_data(y, x, v, coupling=coupling)
    slow = rk4_step(state, 0.01, 0.2, coupling=coupling)
    fast = compiled_step(coupling)(state, 0.01, 0.2, (0.0, 0.0, 0.0))
    np.testing.assert_allclose(fast, slow, atol=1e-15, rtol=1e-14)

def test_constraints_propagate_for_a_small_pulse():
    errors = []
    for n in (161, 321, 641):
        y = np.linspace(-8, 8, n)
        h = y[1] - y[0]
        x = np.zeros((7, n))
        x[4], x[5] = (0.3, 0.01 * np.exp(-y ** 2) * np.cos(2 * y))
        v = np.zeros_like(x)
        v[0], v[4], v[5] = (-0.1, 0.2, -derivatives(x, h)[0][5])
        state = constraint_initial_data(y, x, v)
        run = evolve(y, state, 1.0, 0.4 * h, 2.0)
        errors.append(run.snapshots[-1]['max_null_constraint'])
    assert errors[0] / errors[1] > 10
    assert errors[1] / errors[2] > 12
