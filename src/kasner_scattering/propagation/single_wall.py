"""Numerical equations and integration routines for the local EMS model."""
import itertools
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
TAUB_PI = np.array([-1.0, -1.0, 0.0])

def spectrum(y):
    velocity = y[4:7] - 0.5 * np.sum(y[4:7], axis=0)
    volume = np.sum(velocity, axis=0)
    return np.concatenate((velocity, y[7:8]), axis=0) / volume

def distance(p, q):
    return min((np.linalg.norm(p - np.r_[q[list(perm)], q[3]]) for perm in itertools.permutations(range(3))))

def physical_rhs(t, y):
    omega = np.exp(y[8])
    velocity = y[4:7] - 0.5 * np.sum(y[4:7])
    return np.r_[velocity, y[7], 0.0, 2 * omega, 0.0, -y[3] * omega, -2 * velocity[1] + y[3] * y[7]]

def scaled_rhs(epsilon):

    def rhs(x, u):
        velocity_over_epsilon = np.array([0.0, 0.0, 1 / epsilon]) + u[4:7] - 0.5 * np.sum(u[4:7])
        potential = np.exp(u[8]) / epsilon ** 2
        return np.r_[velocity_over_epsilon, u[7], 0.0, 2 * potential, 0.0, -u[3] * potential, -2 * velocity_over_epsilon[1] + u[3] * u[7]]
    return rhs

def integrate(delta, beta, phi, method, tolerance):
    eps = np.sqrt(delta)
    p = np.array([2, 2, beta ** 2 - 1, 2 * np.sqrt(2) * beta]) / (beta ** 2 + 3)
    initial_omega = 1e-07 * delta
    shift = -2 * initial_omega / (1 + np.sqrt(1 + 3 * initial_omega))
    initial = np.r_[np.zeros(3), phi, p[:3] - 1 + shift, p[3], np.log(initial_omega)]
    independent = method == 'Radau'
    time_unit = eps if independent else 1.0
    u0 = initial.copy()
    if independent:
        u0[4:7] = (initial[4:7] - TAUB_PI) / eps
        u0[7] /= eps
    rhs = scaled_rhs(eps) if independent else physical_rhs

    def physical(u):
        y = u.copy()
        if independent:
            y[4:7] = TAUB_PI.reshape((3,) + (1,) * (u.ndim - 1)) + eps * u[4:7]
            y[7] *= eps
        return y

    def peak_event(t, u):
        return rhs(t, u)[8]
    peak_event.direction = -1
    peak_event.terminal = False

    def end_event(t, u):
        return u[8] - np.log(initial_omega * 1e-08)
    end_event.direction = -1
    end_event.terminal = True
    duration = 160 / (phi * p[3]) * time_unit
    common = dict(method=method, rtol=tolerance, atol=tolerance * 0.01, dense_output=True, max_step=0.35 / (phi * p[3]) * time_unit)
    forward = solve_ivp(rhs, (0.0, duration), u0, events=(peak_event, end_event), **common)
    backward = solve_ivp(rhs, (0.0, -duration), u0, events=end_event, **common)
    if not (forward.success and backward.success and (len(forward.t_events[0]) == 1) and (len(forward.t_events[1]) == 1) and (len(backward.t_events[0]) == 1)):
        raise RuntimeError(f'Incomplete trajectory: {forward.message}; {backward.message}')
    left, right = (backward.t[-1] / time_unit, forward.t[-1] / time_unit)

    def at(t):
        if np.ndim(t) == 0:
            return physical((forward if t >= 0 else backward).sol(t * time_unit))
        return np.stack([at(float(v)) for v in t], axis=-1)
    peak_time = forward.t_events[0][0] / time_unit
    peak_state = at(peak_time)
    peak_log = peak_state[8]
    root_tolerance = 1e-10

    def crossing(log_level, before):
        a, b = (left, peak_time) if before else (peak_time, right)
        return brentq(lambda t: at(t)[8] - log_level, a, b, xtol=root_tolerance)
    fwhm_times = [crossing(peak_log - np.log(2), before) for before in (True, False)]
    volume = lambda t: float(np.sum(at(t)[:3]))
    width = volume(fwhm_times[1]) - volume(fwhm_times[0])

    def shifted(t, amount):
        target = volume(t) + amount
        a, b = (left, t) if amount < 0 else (t, right)
        return brentq(lambda v: volume(v) - target, a, b, xtol=root_tolerance)
    estimates = []
    for level in (1.0, 0.1, 0.01):
        a = crossing(np.log(initial_omega * level), True)
        b = crossing(np.log(initial_omega * level), False)
        for extension in (0, 1):
            ai = shifted(a, -extension * width)
            bi = shifted(b, extension * width)
            ai0, bi1 = (shifted(ai, -width), shifted(bi, width))
            incoming = spectrum(at(np.linspace(ai0, ai, 65)))
            outgoing = spectrum(at(np.linspace(bi, bi1, 65)))
            pin, pout = (incoming.mean(axis=1), outgoing.mean(axis=1))
            reflection = pin * np.array([1, 1, 1, -1])
            error = distance(pout, reflection) / eps
            drift = max(np.linalg.norm(np.ptp(incoming, axis=1)), np.linalg.norm(np.ptp(outgoing, axis=1))) / eps
            endpoints = at(np.array([ai0, ai, bi, bi1]))
            potential = np.exp(endpoints[8])
            scalar_min = np.min(abs(endpoints[7]))
            phi_min = np.min(endpoints[3])
            tail_impulse = 2 * np.max(potential) / (phi_min * scalar_min)
            residual_bound = (4 * tail_impulse + 4 * np.max(potential) / scalar_min) / eps
            estimates.append({'exit_over_initial': level, 'extension_widths': extension, 'in_window': [ai0, ai], 'out_window': [bi, bi1], 'in_spectrum': pin.tolist(), 'out_spectrum': pout.tolist(), 'tangent_error': error, 'platform_drift_tangent': drift, 'tail_effect_bound_tangent': float(residual_bound)})
    sample_t = np.unique(np.r_[np.linspace(left, right, 1601), peak_time, fwhm_times])
    states = at(sample_t)
    pi, pp, omega = (states[4:7], states[7], np.exp(states[8]))
    velocity = pi - 0.5 * pi.sum(axis=0)
    h = 0.5 * ((pi * pi).sum(axis=0) - 0.5 * pi.sum(axis=0) ** 2 + pp ** 2) + omega
    impulse = 0.5 * (states[5] - initial[5])
    a0 = initial[5] - 0.5 * initial[4:7].sum()
    exchange = pp ** 2 + 2 * omega - (initial[7] ** 2 + 2 * initial_omega) + 4 * a0 * impulse + 2 * impulse ** 2
    acceleration = np.vstack((-omega, omega, -omega))
    volume_rate = velocity.sum(axis=0)
    ricci00 = -omega + volume_rate ** 2 - (velocity ** 2).sum(axis=0)
    ricci_spatial = -acceleration
    stress_spatial = np.vstack((omega, -omega, omega))
    einstein = np.vstack((ricci00 - pp ** 2 - omega, ricci_spatial - stress_spatial))
    summary = {'delta': delta, 'epsilon': eps, 'beta_k': beta, 'phi_k': phi, 'method': method, 'rtol': tolerance, 'initial': initial.tolist(), 'nfev': forward.nfev + backward.nfev, 'peak_zeta': float(peak_time), 'peak_log_volume': volume(peak_time), 'peak_q': float(peak_state[3] / 2), 'peak_omega': float(np.exp(peak_log)), 'peak_log_derivative': float(physical_rhs(peak_time, peak_state)[8]), 'fwhm_zeta': fwhm_times, 'fwhm_log_volume': width, 'max_hamiltonian_over_delta': float(np.max(abs(h)) / delta), 'max_exchange_identity_over_delta': float(np.max(abs(exchange)) / delta), 'max_4d_equations_over_delta': float(np.max(abs(einstein)) / delta), 'phi_min': float(states[3].min()), 'estimates': estimates, 'status': 'evaluated; acceptance is in aggregate report'}
    return (summary, {'zeta': sample_t, 'state': states, 'spectrum': spectrum(states), 'hamiltonian': h, 'exchange_identity': exchange, 'einstein': einstein})
