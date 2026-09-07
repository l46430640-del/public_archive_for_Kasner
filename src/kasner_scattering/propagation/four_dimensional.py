"""Numerical equations and integration routines for the local EMS model."""
from functools import lru_cache
import numpy as np
import sympy as sp

@lru_cache(maxsize=1)
def coordinate_tensors():
    r, sigma, p, q, psi, a1, a2, c1, c2, b = x = sp.symbols('x0:10', real=True)
    orbit = sp.exp(r) * sp.Matrix([[sp.exp(p), sp.exp(p) * q], [sp.exp(p) * q, sp.exp(-p) + sp.exp(p) * q ** 2]])
    c = sp.Matrix([c1, c2])
    metric = sp.zeros(4)
    metric[0, 0] = -sp.exp(2 * sigma)
    metric[1, 1] = sp.exp(2 * sigma) + (c.T * orbit * c)[0]
    metric[1, 2:4] = (orbit * c).T
    metric[2:4, 1] = orbit * c
    metric[2:4, 2:4] = orbit
    potential = sp.Matrix([0, b + a1 * c1 + a2 * c2, a1, a2])
    values = sp.Matrix([*metric, *potential])
    jacobian = values.jacobian(x)
    hessians = sp.Matrix([sp.diff(value, xi, xj) for value in values for xi in x for xj in x])
    return tuple((sp.lambdify(x, expr, 'numpy', cse=True) for expr in (values, jacobian, hessians)))

def residuals(position, first, second, coupling='exp_square'):
    """Input ten fields, first[coordinate,field], second[coord,coord,field]."""
    functions = coordinate_tensors()
    values = np.asarray(functions[0](*position), float).reshape(20)
    jac = np.asarray(functions[1](*position), float).reshape(20, 10)
    hess = np.asarray(functions[2](*position), float).reshape(20, 10, 10)
    d = np.einsum('fi,ai->af', jac, first)
    dd = np.einsum('fij,ai,bj->abf', hess, first, first) + np.einsum('fi,abi->abf', jac, second)
    g, dg, ddg = (values[:16].reshape(4, 4), d[:, :16].reshape(4, 4, 4), dd[:, :, :16].reshape(4, 4, 4, 4))
    inv = np.linalg.inv(g)
    dinv = -np.einsum('ab,mbc,cd->mad', inv, dg, inv)
    gamma = np.zeros((4, 4, 4))
    dgamma = np.zeros((4, 4, 4, 4))
    for a in range(4):
        for b in range(4):
            for c in range(4):
                for e in range(4):
                    combination = dg[b, e, c] + dg[c, e, b] - dg[e, b, c]
                    gamma[a, b, c] += inv[a, e] * combination / 2
                    dgamma[:, a, b, c] += (dinv[:, a, e] * combination + inv[a, e] * (ddg[:, b, e, c] + ddg[:, c, e, b] - ddg[:, e, b, c])) / 2
    ricci = np.zeros((4, 4))
    for a in range(4):
        for b in range(4):
            for c in range(4):
                ricci[a, b] += dgamma[c, c, a, b] - dgamma[b, c, a, c]
                for e in range(4):
                    ricci[a, b] += gamma[c, a, b] * gamma[e, c, e] - gamma[e, a, c] * gamma[c, b, e]
    da, dda = (d[:, 16:], dd[:, :, 16:])
    f = da - da.T
    df = dda - dda.transpose(0, 2, 1)
    raised = inv @ f @ inv
    invariant = np.sum(f * raised)
    psi_first, psi_second = (first[:, 4], second[:, :, 4])
    z = np.exp(position[4] ** 2) if coupling == 'exp_square' else 1.0
    zprime = 2 * position[4] * z if coupling == 'exp_square' else 0.0
    source = 2 * np.outer(psi_first, psi_first) + 2 * z * (f @ inv @ f.T - g * invariant / 4)
    einstein = ricci - source
    box_psi = np.sum(inv * (psi_second - np.einsum('cab,c->ab', gamma, psi_first)))
    scalar = box_psi - zprime * invariant / 4
    maxwell = np.zeros(4)
    maxwell_scale = np.zeros(4)
    for mu in range(4):
        draised = dinv[mu] @ f @ inv + inv @ df[mu] @ inv + inv @ f @ dinv[mu]
        log_volume_derivative = np.trace(inv @ dg[mu]) / 2
        term_a = z * draised[mu]
        term_b = (z * log_volume_derivative + zprime * psi_first[mu]) * raised[mu]
        maxwell += term_a + term_b
        maxwell_scale += np.abs(term_a) + np.abs(term_b)
    frame = np.zeros((4, 4))
    frame[0, 0] = np.exp(-position[1])
    frame[1, 1] = np.exp(-position[1])
    frame[2:4, 1] = -position[7:9] * np.exp(-position[1])
    frame[2:4, 2:4] = np.linalg.inv(np.linalg.cholesky(g[2:4, 2:4]).T)
    orthogonal = frame.T @ einstein @ frame
    curvature_scale = max(np.max(np.abs(frame.T @ source @ frame)), np.max(np.abs(frame.T @ ricci @ frame)), 1e-30)
    scalar_scale = np.exp(-2 * position[1]) * (first[0, 0] + first[0, 1]) ** 2
    return {'einstein': orthogonal, 'scalar': float(scalar), 'maxwell': maxwell, 'scalar_over_expansion_squared': float(scalar / scalar_scale), 'maxwell_relative': float(np.max(np.abs(maxwell)) / max(np.max(maxwell_scale), 1e-300)), 'einstein_relative': float(np.max(np.abs(orthogonal)) / curvature_scale)}

def homogeneous_jets(position, velocity, auxiliary, charges=(0.0, 0.0, 0.0), coupling='exp_square'):
    """Use evolution equations only to supply jets, not to define residuals."""
    from .conformal import auxiliary_velocity, nonlinear_force, symbolic_system
    system = symbolic_system(coupling)
    jac_fn = sp.lambdify((*system['fields'], *system['charges']), system['auxiliary'].jacobian(system['fields']), 'numpy', cse=True)
    aux_v = auxiliary_velocity(position, charges, coupling)
    aux_acc = np.asarray(jac_fn(*position, *charges), float) @ velocity
    first, second = (np.zeros((4, 10)), np.zeros((4, 4, 10)))
    first[0] = np.r_[velocity, aux_v]
    second[0, 0] = np.r_[nonlinear_force(position, velocity, np.zeros(7), charges, coupling), aux_acc]
    return (np.r_[position, auxiliary], first, second)
