import numpy as np
from kasner_scattering.propagation.directed import combine, energy_coefficients, grid_for, normalization, protocol, record_for_delta
from kasner_scattering.propagation.response_model import profiles
from kasner_scattering.propagation.scales import scale_record

def test_new_background_not_reference_arrays():
    record = record_for_delta()
    assert record['background']['delta_target'] == 1e-05
    old = record_for_delta(1e-06)
    assert record['polarization'] == old['polarization']
    assert not np.array_equal(profiles(record, np.array([0.0, 1.0])), profiles(old, np.array([0.0, 1.0])))

def test_equal_energy_at_new_background():
    energy = energy_coefficients()
    refined = energy_coefficients(nodes=4096)
    assert np.max(abs(energy - refined)) / energy[0] < 1e-10
    assert energy[0] != energy_coefficients(1e-06)[0]
    for mix in protocol()['mixing'].values():
        assert abs(normalization(mix) ** 2 * np.polynomial.polynomial.polyval(mix, energy) / energy[0] - 1) < 1e-14

def test_cross_term_only_omitted_in_diagnostic():
    coefs = np.ones((4, 3, 7))
    mix = 2e-05
    assert np.allclose(combine(coefs, mix) - combine(coefs, mix, coherent=False), normalization(mix) ** 2 * mix, rtol=1e-10, atol=1e-15)

def test_observation_domain_and_grid_refinement():
    scale = scale_record(record_for_delta())
    end = 1.4 * scale['frozen_unit_wall_clock_T']
    coarse, radius = grid_for(128, end)
    fine, fine_radius = grid_for(256, end)
    assert radius == fine_radius == scale['packet_width'] / 4
    assert np.array_equal(coarse, fine[::2])
    assert coarse[-1] - radius > end
    assert np.diff(coarse)[0] * scale['k'] < 0.6

def test_new_nonlinear_initial_constraints_match_order_two():
    from kasner_scattering.propagation.directed_nonlinear import nonlinear_initial
    from kasner_scattering.propagation.response_model import initial_expansion
    grid, _ = grid_for(128, 55.0)
    bg, expansion, _ = initial_expansion(record_for_delta(), grid)
    for mixing in protocol()['mixing'].values():
        state, _ = nonlinear_initial(grid, mixing)
        weights = np.array([1.0, mixing, mixing ** 2]) * normalization(mixing) ** 2
        assert np.max(abs(state[1] - weights @ expansion[[5, 13, 21]])) < 1e-16
        assert np.max(abs(state[8] - bg[4] - weights @ expansion[[9, 17, 25]])) < 1e-13

def test_carrier_phase_width_identity_and_soft_momentum():
    for delta in (1e-05, 1e-06):
        record = record_for_delta(delta)
        scale = scale_record(record)
        assert np.isclose(scale['k'] * scale['packet_width'], 0.2 / np.sqrt(delta), rtol=1e-14)
        fields = profiles(record, np.linspace(-scale['packet_width'], scale['packet_width'], 33))
        assert np.array_equal(fields[4], -fields[5])

def test_time_space_interpolation_preserves_linear_profiles():
    from kasner_scattering.propagation.directed_analysis import sample
    t = np.arange(5, dtype=float)
    y = np.arange(-3, 4, dtype=float)
    coefs = t[:, None, None] + np.arange(3)[None, :, None] + 2 * y[None, None, :]
    wanted_t, wanted_y = (np.array([0.5, 2.2]), np.array([-0.7, 1.9]))
    expected = wanted_t[:, None, None] + np.arange(3)[None, :, None] + 2 * wanted_y[None, None, :]
    assert np.allclose(sample(coefs, t, y, wanted_t, wanted_y), expected, atol=1e-14)
