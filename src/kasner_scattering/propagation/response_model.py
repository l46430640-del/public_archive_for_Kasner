"""Numerical equations and integration routines for the local EMS model."""
from functools import lru_cache
import json
import numpy as np
from numba import njit
from scipy.integrate import cumulative_simpson
from scipy.special import roots_legendre
from .conformal import homogeneous_constraint_velocity
from .inputs import HERE
from .scales import kasner_parameters, projected_components, scale_record

def reference_record():
    return json.loads((HERE / 'matching_data.json').read_text())['records'][-1]

def background_initial(record):
    ps, _, v = kasner_parameters(record)
    x, velocity = (np.zeros(7), np.zeros(7))
    x[4], velocity[0], velocity[4] = (record['background']['psi'], -2 * ps, v)
    b = record['background']
    charge = 4 * b['charge'] * np.exp(2 * b['sigma_k']) / b['ktrace']
    velocity = homogeneous_constraint_velocity(x, velocity, (charge, 0.0, 0.0))
    return (np.array([x[0], x[1], x[4], velocity[0], velocity[1], velocity[4]]), charge)

def profiles(record, grid, phase=0.0):
    """Analytic carrier plus a right-moving smooth soft packet, before scaling."""
    scale, components = (scale_record(record), projected_components(record))
    width, k, amplitude = (scale['packet_width'], scale['k'], scale['packet_amplitude'])
    u = np.asarray(grid) / width
    f, fy = (np.zeros_like(u), np.zeros_like(u))
    inside = abs(u) < 1
    f[inside] = np.exp(1 - 1 / (1 - u[inside] ** 2))
    fy[inside] = -2 * u[inside] * f[inside] / (width * (1 - u[inside] ** 2) ** 2)
    rotation = 1.0 if phase == 0 else 1j if phase == np.pi / 2 else np.exp(1j * phase)
    c = components['soft_magnetic'] / (-1j * k)
    carrier = rotation * np.exp(-1j * k * np.asarray(grid))
    low_k = k / 100
    soft = rotation * np.exp(-1j * low_k * np.asarray(grid))
    uc = amplitude * np.real(c * carrier) * f
    ec = -amplitude * np.real(components['transverse_electric'] * carrier) * f
    bc = amplitude * np.real(c * carrier * (fy - 1j * k * f))
    ul = amplitude * abs(c) * np.real(soft) * f
    bl = amplitude * abs(c) * np.real(soft * (fy - 1j * low_k * f))
    el = -bl
    return np.array([uc, ec, bc, ul, el, bl])

@lru_cache(maxsize=8)
def energy_coefficients(phase=0.0, nodes=2048):
    record = reference_record()
    width = scale_record(record)['packet_width']
    points, weights = roots_legendre(nodes)
    _, e, b, _, el, bl = profiles(record, width * points, phase)
    return 2 * width * np.array([np.dot(weights, e * e + b * b), 2 * np.dot(weights, e * el + b * bl), np.dot(weights, el * el + bl * bl)])

def normalization(mixing, phase=0.0):
    energy = energy_coefficients(phase)
    target = energy_coefficients(0.0)[0]
    return np.sqrt(target / np.polynomial.polynomial.polyval(mixing, energy))

def initial_expansion(record, grid, phase=0.0, fixed_kasner=False):
    bg, charge = background_initial(record)
    if fixed_kasner:
        ps, pt, _ = kasner_parameters(record)
        bg[4] = -pt
        charge = 0.0
    uc, ec, _, ul, el, _ = profiles(record, grid, phase)
    h = grid[1] - grid[0]
    from .evolution import derivatives
    bc, _ = derivatives(uc[None], h)
    bl, _ = derivatives(ul[None], h)
    bc, bl = (bc[0], bl[0])
    g = bg[2] * bg[5]
    fields = np.zeros((28, len(grid)))
    fields[:4] = [uc, ec + g * uc, ul, el + g * ul]
    beta = charge ** 2 * np.exp(-bg[2] ** 2 - 2 * bg[0] + 2 * bg[1]) / 8
    for index, u2, en, flux in ((0, uc * uc, ec * ec + bc * bc, ec * bc), (1, 2 * uc * ul, 2 * (ec * el + bc * bl), ec * bl + el * bc), (2, ul * ul, el * el + bl * bl, el * bl)):
        base = 4 + 8 * index
        sigma_y = 2 * np.exp(-bg[0]) * flux / bg[3]
        primitive = cumulative_simpson(sigma_y, x=grid, initial=0.0)
        sigma = primitive - primitive[len(grid) // 2]
        fields[base + 1] = sigma
        fields[base + 5] = (beta * sigma + 2 * beta * np.exp(-bg[0]) * u2 + np.exp(-bg[0]) * en) / bg[3]
    return (bg, fields, charge)

@njit
def background_rhs(bg, charge):
    r, sigma, psi, rt, st, pt = bg
    beta = charge ** 2 * np.exp(-psi * psi - 2 * r + 2 * sigma) / 8
    return np.array([rt, st, pt, beta - rt * rt, -beta + rt * rt / 4 - pt * pt, psi * beta / 2 - rt * pt])

@njit
def fd_row(state, row, j, h):
    if j < 2 or j >= state.shape[1] - 2:
        return (0.0, 0.0)
    c = state[row, j]
    first = (8 * (state[row, j + 1] - state[row, j - 1]) - (state[row, j + 2] - state[row, j - 2])) / (12 * h)
    second = (16 * (state[row, j - 1] - c + (state[row, j + 1] - c)) - (state[row, j - 2] - c) - (state[row, j + 2] - c)) / (12 * h * h)
    return (first, second)

@njit
def response_rhs(bg, state, h, charge):
    db = background_rhs(bg, charge)
    r, sigma, psi, rt, st, pt = bg
    beta = charge ** 2 * np.exp(-psi * psi - 2 * r + 2 * sigma) / 8
    er, g = (np.exp(-r), psi * pt)
    pump = pt * pt + psi * db[5] + g * g - 2 * beta
    out = np.zeros_like(state)
    for j in range(state.shape[1]):
        b0, d20 = fd_row(state, 0, j, h)
        b1, d21 = fd_row(state, 2, j, h)
        u0, u1 = (state[0, j], state[2, j])
        e0, e1 = (state[1, j] - g * u0, state[3, j] - g * u1)
        out[0, j], out[2, j] = (state[1, j], state[3, j])
        out[1, j], out[3, j] = (d20 + pump * u0, d21 + pump * u1)
        for index in range(3):
            if index == 0:
                u2, difference = (u0 * u0, e0 * e0 - b0 * b0)
            elif index == 1:
                u2, difference = (2 * u0 * u1, 2 * (e0 * e1 - b0 * b1))
            else:
                u2, difference = (u1 * u1, e1 * e1 - b1 * b1)
            i = 4 + 8 * index
            dr, ds, dp, dpsi = state[i:i + 4, j]
            drt, dst, dpt, dpsit = state[i + 4:i + 8, j]
            delta_beta = beta * (-2 * psi * dpsi - 2 * dr + 2 * ds)
            source = np.array([-2 * rt * drt + delta_beta + 4 * beta * er * u2, rt * drt / 2 - 2 * pt * dpsit - delta_beta - 6 * beta * er * u2, -rt * dpt - 2 * er * difference + 4 * beta * er * u2, -rt * dpsit - pt * drt + beta * dpsi / 2 + psi * delta_beta / 2 + psi * er * difference])
            for n in range(4):
                _, second = fd_row(state, i + n, j, h)
                out[i + n, j] = state[i + n + 4, j]
                out[i + n + 4, j] = second + source[n]
    return (db, out)

@njit
def response_step(bg, state, h, dt, charge):
    b1, k1 = response_rhs(bg, state, h, charge)
    b2, k2 = response_rhs(bg + dt * b1 / 2, state + dt * k1 / 2, h, charge)
    b3, k3 = response_rhs(bg + dt * b2 / 2, state + dt * k2 / 2, h, charge)
    b4, k4 = response_rhs(bg + dt * b3, state + dt * k3, h, charge)
    return (bg + dt * (b1 + 2 * b2 + 2 * b3 + b4) / 6, state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6)

def scalar_coefficients(bg, state):
    trace = bg[3] + bg[4]
    p0 = np.sqrt(2) * bg[5] / abs(trace)
    output = []
    for i in (4, 12, 20):
        output.append(np.sqrt(2) / abs(trace) * (state[i + 7] - bg[5] * (state[i + 4] + state[i + 5]) / trace))
    return (p0, np.array(output))

def prediction(coefficients, mixing, phase=0.0):
    return normalization(mixing, phase) ** 2 * np.einsum('i,...ij->...j', [1.0, mixing, mixing ** 2], coefficients)
