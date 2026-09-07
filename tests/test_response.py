import numpy as np
from kasner_scattering.propagation.conformal import nonlinear_force
from kasner_scattering.propagation.response_model import background_initial, background_rhs, energy_coefficients, initial_expansion, normalization, prediction, reference_record, response_rhs
from kasner_scattering.propagation.response_experiment import nonlinear_initial

def test_charged_background_uses_full_action():
    bg, q = background_initial(reference_record())
    x, v = (np.zeros(7), np.zeros(7))
    x[[0, 1, 4]], v[[0, 1, 4]] = (bg[:3], bg[3:])
    direct = nonlinear_force(x, v, np.zeros(7), (q, 0.0, 0.0))
    assert np.allclose(direct[[0, 1, 4]], background_rhs(bg, q)[3:], rtol=1e-13, atol=1e-17)

def test_prediction_handles_time_and_space_axes():
    coefficients = np.ones((7, 3, 5))
    assert prediction(coefficients, 0.2).shape == (7, 5)
    assert np.allclose(prediction(coefficients, 0.2)[0], prediction(coefficients[0], 0.2))

def test_energy_normalization_continuum():
    for phase in (0.0, np.pi / 2):
        a, b = (energy_coefficients(phase, 2048), energy_coefficients(phase, 4096))
        assert np.max(abs(a - b)) / a[0] < 1e-10
        for mixing in (0.0, 1e-06, 0.2):
            value = normalization(mixing, phase) ** 2 * np.polynomial.polynomial.polyval(mixing, a)
            assert abs(value / energy_coefficients()[0] - 1) < 1e-14

def test_second_order_force_matches_taylor_of_full_action():
    bg = np.array([0.12, -0.2, 0.6, -0.08, -0.3, 0.17])
    charge = 0.4
    state = np.zeros((28, 5))
    state[0], state[1] = (0.3, -0.1)
    state[4:12] = np.array([0.1, -0.04, 0.07, 0.08, 0.02, 0.03, -0.01, 0.04])[:, None]
    _, response = response_rhs(bg, state, 1.0, charge)
    coefficients = response[[8, 9, 10, 11], 2]
    db = background_rhs(bg, charge)
    errors = []
    for epsilon in (0.02, 0.01, 0.005):
        x, v = (np.zeros(7), np.zeros(7))
        x[[0, 1, 4]], v[[0, 1, 4]] = (bg[:3], bg[3:])
        x[[0, 1, 2, 4]] += epsilon ** 2 * state[4:8, 2]
        v[[0, 1, 2, 4]] += epsilon ** 2 * state[8:12, 2]
        x[5] = epsilon * state[0, 2] * np.exp(-bg[2] ** 2 / 2)
        v[5] = epsilon * (state[1, 2] - bg[2] * bg[5] * state[0, 2]) * np.exp(-bg[2] ** 2 / 2)
        direct = nonlinear_force(x, v, np.zeros(7), (charge, 0.0, 0.0))[[0, 1, 2, 4]]
        background = np.array([db[3], db[4], 0.0, db[5]])
        errors.append(np.max(abs((direct - background) / epsilon ** 2 - coefficients)))
    assert min(np.log2(np.array(errors[:-1]) / errors[1:])) > 1.95

def test_gradient_source_matches_full_action_and_linear_twist_term():
    bg = np.array([0.12, -0.2, 0.6, -0.08, -0.3, 0.17])
    charge = 0.4
    grid = np.arange(-2, 3, dtype=float)
    state = np.zeros((28, 5))
    state[0] = 0.3 + 0.07 * grid
    state[1] = -0.1 + 0.02 * grid
    _, response = response_rhs(bg, state, 1.0, charge)
    source = response[[8, 9, 10, 11], 2]
    db = background_rhs(bg, charge)
    epsilon = 0.001
    x, v, w = (np.zeros(7), np.zeros(7), np.zeros(7))
    x[[0, 1, 4]], v[[0, 1, 4]] = (bg[:3], bg[3:])
    rootz = np.exp(bg[2] ** 2 / 2)
    x[5] = epsilon * 0.3 / rootz
    v[5] = epsilon * (-0.1 - bg[2] * bg[5] * 0.3) / rootz
    w[5] = epsilon * 0.07 / rootz
    direct = nonlinear_force(x, v, w, (charge, 0.0, 0.0))
    base = np.array([db[3], db[4], 0.0, db[5]])
    assert np.max(abs((direct[[0, 1, 2, 4]] - base) / epsilon ** 2 - source)) < 1e-09
    g = bg[2] * bg[5]
    gt = bg[5] ** 2 + bg[2] * db[5]
    transformed = rootz * direct[5] / epsilon + 2 * g * (-0.1 - g * 0.3) + (gt + g * g) * 0.3
    assert abs(transformed - response[1, 2]) < 1e-13

def test_interference_sources_are_coherent():
    bg, charge = background_initial(reference_record())
    grid = np.arange(-2, 3, dtype=float)
    state = np.zeros((28, 5))
    state[0], state[1] = (0.3 + 0.02 * grid, 0.05)
    state[2], state[3] = (-0.1 + 0.07 * grid, 0.03)
    _, original = response_rhs(bg, state, 1.0, charge)
    mixing = 0.7
    combined = np.zeros_like(state)
    combined[0] = state[0] + mixing * state[2]
    combined[1] = state[1] + mixing * state[3]
    _, total = response_rhs(bg, combined, 1.0, charge)
    expected = original[4:12] + mixing * original[12:20] + mixing ** 2 * original[20:28]
    assert np.allclose(total[4:12], expected, rtol=1e-13, atol=1e-15)

def test_initial_constraints_have_correct_quadratic_expansion():
    grid = np.arange(-700, 701) * 0.4
    record = reference_record()
    bg, expansion, _ = initial_expansion(record, grid)
    for mixing in (0.0, 0.001):
        initial, _ = nonlinear_initial(grid, mixing)
        coeff = np.array([1.0, mixing, mixing ** 2]) * normalization(mixing) ** 2
        sigma = coeff @ expansion[[5, 13, 21]]
        sigmat = coeff @ expansion[[9, 17, 25]]
        assert np.max(abs(initial[1] - sigma)) < 1e-18
        assert np.max(abs(initial[8] - bg[4] - sigmat)) < 2e-15
