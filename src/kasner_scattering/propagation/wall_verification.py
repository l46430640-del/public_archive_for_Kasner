import json
from pathlib import Path
import numpy as np
from .inputs import ROOT

def verify_wall(directory, write_outputs=True):
    OUT = Path(directory)
    if write_outputs and (ROOT / 'build').resolve() not in OUT.resolve().parents:
        raise ValueError('verification outputs must be strictly below build/')
    HERE = OUT
    records = [json.loads(path.read_text()) for path in OUT.glob('d*.json')]
    rows = []
    for delta in (1e-06, 1e-08):
        for phi in (10, 20, 40, 80):
            selected = [r for r in records if r['delta'] == delta and r['phi_k'] == phi]
            assert len(selected) == 4
            fine = next((r for r in selected if r['rtol'] == 1e-13))
            ref = fine['estimates'][-1]

            def difference(e):
                return sum((np.linalg.norm(np.array(e[key]) - ref[key]) for key in ('in_spectrum', 'out_spectrum'))) / np.sqrt(delta)
            numeric = max((difference(r['estimates'][-1]) for r in selected))
            window = max((difference(e) for e in fine['estimates']))
            drift = max((e['platform_drift_tangent'] for e in fine['estimates']))
            residual = max((e['tail_effect_bound_tangent'] for e in fine['estimates']))
            error = ref['tangent_error']
            raw = np.load(OUT / f'd{delta:.0e}_phi{phi}_DOP853_r1e-13.npz')
            t = raw['zeta']
            uniform = np.linspace(t[0], t[-1], 1601)
            ids = [np.argmin(abs(t - x)) for x in uniform]
            states = raw['state'][:, ids].astype(np.longdouble)
            checks = []
            for stride in (4, 2, 1):
                y = states[:, ::stride]
                h = np.longdouble(uniform[1] - uniform[0]) * stride
                derivative = lambda v: (v[..., :-4] - 8 * v[..., 1:-3] + 8 * v[..., 3:-1] - v[..., 4:]) / (12 * h)
                pi = y[4:7]
                velocity = pi - 0.5 * pi.sum(axis=0)
                acceleration = derivative(velocity)
                pp = y[7, 2:-2]
                omega = np.exp(y[8, 2:-2])
                u_prime = derivative(y[7])
                v = velocity[:, 2:-2]
                ricci00 = acceleration.sum(axis=0) + v.sum(axis=0) ** 2 - (v * v).sum(axis=0)
                spatial_source = np.vstack((omega, -omega, omega))
                einstein = np.vstack((ricci00 - pp * pp - omega, -acceleration - spatial_source))
                scalar = -u_prime - y[3, 2:-2] * omega
                magnetic = derivative(y[8]) - (-2 * v[1] + y[3, 2:-2] * pp)
                checks.append({'stride': stride, 'Einstein_over_delta': float(np.max(abs(einstein)) / delta), 'scalar_over_delta': float(np.max(abs(scalar)) / delta), 'magnetic_log_equation': float(np.max(abs(magnetic))), 'scope': 'fourth-order derivative of stored geometric velocities; no RHS acceleration substitution'})
            row = {'delta': delta, 'phi_k': phi, 'q_B': fine['peak_q'], 'tangent_error': error, 'q_times_error': fine['peak_q'] * error, 'full_spectrum_numerical_difference': numeric, 'full_spectrum_window_difference': window, 'platform_drift': drift, 'finite_window_residual_bound': residual, 'peak_q_difference': max((abs(r['peak_q'] - fine['peak_q']) for r in selected)), 'max_H_over_delta': max((r['max_hamiltonian_over_delta'] for r in selected)), 'max_exact_exchange_over_delta': max((r['max_exchange_identity_over_delta'] for r in selected)), 'direct_4d_difference_checks': checks, 'accepted': bool(max(numeric, window, drift, residual) < error / 10)}
            rows.append(row)
    report = {'all_eight_accepted': all((r['accepted'] for r in rows)), 'rows': rows, 'normalization': 'd_K(P_out,R_infinity P_in)/sqrt(delta), same trajectory and geometric volume normalization', 'qualification': 'finite single-axis controls, not black-hole matching or a proof inferred from a trend'}
    if not write_outputs:
        return report
    (OUT / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
    table = []
    for r in rows:
        exponent = int(round(np.log10(r['delta'])))
        uncertainty = max((r[k] for k in ('full_spectrum_numerical_difference', 'full_spectrum_window_difference', 'platform_drift', 'finite_window_residual_bound')))
        table.append(f"$10^{{{exponent}}}$ & {r['phi_k']} & {r['q_B']:.3f} & {r['tangent_error']:.3f} & {r['q_times_error']:.3f} & {100000000.0 * uncertainty:.2f} \\")
    (HERE / 'wall_rows.tex').write_text('\n'.join((v + '\\' for v in table)) + '\n')
    print(json.dumps({'all_eight_accepted': report['all_eight_accepted'], 'max_full_spectrum_numeric': max((r['full_spectrum_numerical_difference'] for r in rows)), 'max_full_spectrum_window': max((r['full_spectrum_window_difference'] for r in rows)), 'max_drift_or_residual': max((max(r['platform_drift'], r['finite_window_residual_bound']) for r in rows)), 'q_times_error_range': [min((r['q_times_error'] for r in rows)), max((r['q_times_error'] for r in rows))], 'fine_4d_residuals': [r['direct_4d_difference_checks'][-1] for r in rows]}, indent=2))
    return report
