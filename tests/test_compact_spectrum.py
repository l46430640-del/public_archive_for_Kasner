import numpy as np
from scipy.integrate import simpson
from kasner_scattering.propagation.compact_spectrum import derivative_norm, transform, transform_mp

def test_compact_fourier_carrier_tail_is_not_roundoff():
    frequencies = np.array([0.0, 197.0, 200.0, 203.0])
    actual = transform(frequencies, 1024)
    expected = np.array([float(transform_mp(value, 40)) for value in frequencies])
    np.testing.assert_allclose(actual, expected, atol=2e-14, rtol=2e-12)
    assert abs(expected[2]) > 1e-09

def test_derivative_norms_have_upward_margin():
    for order in (4, 6):
        result = derivative_norm(order)
        assert result['upward_rounded_norm'] > result['norm_estimate']
        assert result['interior_critical_points'] > order

def test_compact_inverse_transform_includes_envelope_derivative():
    xi = np.arange(-1200, 1201) * 0.5
    k, width, y = (1.43, 140.0, 23.0)
    c, electric = (0.2 + 0.7j, -0.8 + 0.1j)
    q = k + xi / width
    weight = transform(xi, 1024) / (2 * np.pi)
    carrier = np.exp(-1j * q * y)
    fields = np.array([simpson(electric * carrier * weight, x=xi).real, simpson(c * (-1j * q) * carrier * weight, x=xi).real])
    u = y / width
    profile = np.exp(1 - 1 / (1 - u * u)) * np.exp(-1j * k * y)
    derivative = -2 * u / (width * (1 - u * u) ** 2)
    expected = np.array([(electric * profile).real, (c * (derivative - 1j * k) * profile).real])
    np.testing.assert_allclose(fields, expected, atol=2e-07, rtol=0.0)
