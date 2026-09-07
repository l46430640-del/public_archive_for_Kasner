"""Numerical equations and integration routines for the local EMS model."""
import numpy as np
from scipy.integrate import simpson, trapezoid
from scipy.signal import find_peaks

def window_metrics(data, point, left, right):
    sl = slice(left, right + 1)
    times = data['times'][sl]
    spec = data['spectrum'][sl, point]
    metrics = {'start_time': float(times[0]), 'end_time': float(times[-1]), 'duration_volume': float(data['volume'][right, point] - data['volume'][left, point]), 'drift': float(np.max(np.ptp(spec, axis=0))), 'mean_spectrum': np.mean(spec, axis=0).tolist(), 'max_em_fraction': float(np.max(data['magnetic'][sl, point] + data['electric'][sl, point])), 'max_spatial_fraction': float(np.max(data['spatial_fraction'][sl, point])), 'max_axis_error': float(np.max(data['axis_error'][sl, point]))}
    for key in ('matter_rate', 'spatial_rate', 'vacuum_rate'):
        rates = data[key][sl, point]
        metrics[key + '_absolute_integral'] = float(np.max(trapezoid(abs(rates), times, axis=0)))
        metrics[key + '_signed_integral'] = trapezoid(rates, times, axis=0).tolist()
        metrics[key + '_quadrature_difference'] = float(np.max(abs(simpson(abs(rates), x=times, axis=0) - trapezoid(abs(rates), times, axis=0))))
    return metrics

def admissible(metrics, tolerance):
    return all((metrics[key] <= tolerance for key in ('drift', 'max_em_fraction', 'max_spatial_fraction', 'matter_rate_absolute_integral', 'spatial_rate_absolute_integral'))) and metrics['max_axis_error'] < tolerance / 10

def required_tolerance(metrics):
    return max((metrics[key] for key in ('drift', 'max_em_fraction', 'max_spatial_fraction', 'matter_rate_absolute_integral', 'spatial_rate_absolute_integral')))

def first_event(data, point, tolerance=0.0001):
    time, volume = (data['times'], data['volume'][:, point])
    if np.any(np.diff(volume) <= 0):
        return {'status': 'NONMONOTONE_VOLUME_NO_CLOCK'}
    magnetic = data['magnetic'][:, point]
    threshold = 0.01 * data['spectrum'][0, point, 3] ** 2
    peaks = find_peaks(magnetic, height=threshold)[0]
    if not len(peaks):
        return {'status': 'NO_RESOLVED_PEAK_IN_WINDOW', 'final_magnetic': float(magnetic[-1]), 'max_magnetic': float(max(magnetic)), 'threshold': float(threshold)}
    peak = peaks[0]
    left, right = (peak, peak)
    while left > 0 and magnetic[left] >= magnetic[peak] / 2:
        left -= 1
    while right < len(time) - 1 and magnetic[right] >= magnetic[peak] / 2:
        right += 1
    if left == 0 or right == len(time) - 1:
        return {'status': 'FIRST_PULSE_WIDTH_NOT_CLOSED', 'peak_time': float(time[peak])}
    width = volume[right] - volume[left]
    polynomial = np.polyfit(time[peak - 1:peak + 2] - time[peak], magnetic[peak - 1:peak + 2], 2)
    vertex = float(time[peak] - polynomial[1] / (2 * polynomial[0]))
    output = {'status': 'RESOLVED_MAGNETIC_PULSE_NOT_YET_SCATTERING', 'position': float(data['points'][point]), 'peak_time': float(time[peak]), 'peak_volume_clock': float(volume[peak] - volume[0]), 'peak_fraction': float(magnetic[peak]), 'quadratic_peak_time': vertex, 'quadratic_peak_volume_clock': float(np.interp(vertex, time, volume) - volume[0]), 'peak_time_bracket': [float(time[peak - 1]), float(time[peak + 1])], 'half_maximum_times': [float(time[left]), float(time[right])], 'width_volume': float(width), 'samples_across_width': int(right - left), 'peak_spectrum': data['spectrum'][peak, point].tolist(), 'initial_spectrum': data['spectrum'][0, point].tolist(), 'final_spectrum': data['spectrum'][-1, point].tolist()}

    def candidate_windows(begin, stop):
        result = []
        for a in range(begin, stop):
            b = int(np.searchsorted(volume, volume[a] + width))
            if b >= stop:
                break
            metrics = window_metrics(data, point, a, b)
            result.append(metrics)
        return result
    before = [m for m in candidate_windows(0, left + 1) if admissible(m, tolerance)]
    all_after = candidate_windows(right, len(time))
    after = [m for m in all_after if admissible(m, tolerance)]
    output['incoming_candidate'] = before[-1] if before else None
    output['outgoing_candidate'] = after[0] if after else None
    best = min(all_after, key=required_tolerance) if all_after else None
    output['best_post_event_window'] = best
    output['minimum_required_tolerance'] = required_tolerance(best) if best else None
    tail_start = int(np.searchsorted(volume, volume[-1] - width))
    output['last_width_diagnostics'] = window_metrics(data, point, tail_start, len(time) - 1)
    output['interaction_integrals'] = window_metrics(data, point, left, right)
    output['platform_accepted'] = False
    output['p_plus'], output['scattering_error'] = (None, None)
    return output
