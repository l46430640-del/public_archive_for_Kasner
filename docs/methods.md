# Methods

## Equations

We use signature (-,+,+,+) and
\[
 I=(16\pi)^{-1}\int\sqrt{-g}\,[R-2(\nabla\psi)^2-ZF^2]\,d^4x,
 \qquad Z=\exp(\psi^2).
\]
The control sets \(Z=1\). The canonical scalar is \(\varphi=\sqrt2\psi\).
On contracting slices \(K_{ij}=\mathcal L_n\gamma_{ij}/2\), and
\[
 P=\left(\operatorname{eig}(K^i{}_j/\operatorname{tr}K);
 \sqrt2\,n(\psi)/|\operatorname{tr}K|\right).
\]
Only spatial eigenvalues are permuted.

The two-Killing metric has conformal base \(e^{2\sigma}(-dT^2+dy^2)\),
orbit metric \(\rho S\), and
\[
 \rho=e^r,\quad S=\begin{pmatrix}e^P&e^PQ\\e^PQ&e^{-P}+e^PQ^2\end{pmatrix}.
\]
In the sector \(F_{AB}=0\), the Routhian for
\(X=(r,\sigma,P,Q,\psi,a_1,a_2)\) is
\[
 \mathcal L=\tfrac12X_T^t\mathsf KX_T-\tfrac12X_y^t\mathsf KX_y-V.
\]
Nonzero symmetric kinetic entries are
\(\mathsf K_{rr}=-\rho\), \(\mathsf K_{r\sigma}=-2\rho\),
\(\mathsf K_{PP}=\rho\), \(\mathsf K_{QQ}=\rho e^{2P}\),
\(\mathsf K_{\psi\psi}=4\rho\), and
\(\mathsf K_{a_Aa_B}=4ZS^{AB}\). The potential is
\[
 V=e^{2\sigma}\left[\frac{q^2}{8\rho Z}
 +\frac{(J-qa)^tS^{-1}(J-qa)}{2\rho^2}\right].
\]
Electric and orbit momenta \(q,J_A\) are constant. Variation of the base
before gauge fixing retains two constraints. The conformal and evolution
modules construct these equations and solve the initial constraints,
including Maxwell momentum flux.

## Coherent prediction

The background parameter is \(\delta=(Q_{\rm BH}/M)/q_c-1\).
Measured interior parameters give
\[
 p_s=2/(\beta^2+3),\quad p_\parallel=(\beta^2-1)/(\beta^2+3),
 \quad p_\psi=2\sqrt2\beta/(\beta^2+3).
\]
The existing sequence supports \(\beta\sqrt\delta\) approaching approximately
0.076817; no new fit is used. For
\(\Lambda=a_\parallel|\operatorname{tr}K|\) on the input surface,
\[
 k=\omega/\Lambda,\quad w=\Lambda M\tau_0/\sqrt\delta,\quad
 kw=(\omega M)\tau_0/\sqrt\delta.
\]
Width and complex transmitted phase both change with background.
The free-flight unit-wall threshold is a proxy. Vanishing peak energy
contributes to the actual peak clock and prevents identifying its
asymptotic coefficient with that proxy.

The zero-packet solution has fields \(r_0,\sigma_0,\psi_0\).
Let \(U=e^{\psi_0^2/2}a_1\), \(g=\psi_0\psi_{0,T}\),
\(b=(q^2/8)e^{-\psi_0^2-2r_0+2\sigma_0}\). Then
\[
 U_{TT}-U_{yy}=(g_T+g^2-2b)U,\quad e=U_T-gU,\quad m=U_y.
\]
The signed source is \(e^2-m^2\). Propagated quadratic corrections
\(R,\Sigma,D,\Phi\) give
\[
 \Delta p_\psi^{(2)}=\frac{\sqrt2}{|\mathcal K_0|}
 [\Phi_T-\psi_{0,T}(R_T+\Sigma_T)/\mathcal K_0],
 \quad\mathcal K_0=r_{0,T}+\sigma_{0,T}.
\]
For \(U=N(U_c+\lambda U_\ell)\), the response is
\(N^2(C_{cc}+\lambda C_{c\ell}+\lambda^2C_{\ell\ell})\).
The mixed coefficient includes the factor of two in the signed source.
The full initial energy quadratic form fixes \(N\). Its value is
\(2\int(e^2+m^2)\,dy\) at \(r=P=0\), a reduced electromagnetic energy.
The scalar receives \(4\psi_0\psi_{0,T}(e^2-m^2)\); Maxwell receives its
opposite. Geometry and background fields supply the other work terms.

## Integration

The main solver uses fourth-order spatial differences and RK4 with no
periodic wrap. Its domain covers the causal past of the observation region.
Independent interaction calculations use sixth-order differences and
DOP853, with checkpoint scope specified in [the evidence map](evidence.md).
Exchange windows retain the stored magnetic peak, half-height endpoints
and integral weights. Platform tests extend the observation window and
require at least one measured event width.

Reconstruction fixes and checks one active BLAS thread through
[threadpoolctl](https://github.com/joblib/threadpoolctl/tree/3.6.0).
This setting controls arithmetic ordering in stencil construction and adaptive
integration. Derivative-fit residuals describe the specified calculations;
solution errors are assessed separately by spatial refinement, time-step
changes and independent discretization.

## Magnetic reference

The self-consistent homogeneous reference has one magnetic axis (the second
physical axis) and zero electric fields:
\[
 H=\tfrac12\sum_a\pi_a^2-\tfrac14(\sum_a\pi_a)^2+\tfrac12u^2+\Omega_B=0,
 \quad \Omega_B=B^2e^{-2\beta_2+\varphi^2/2},\quad u=\pi_\varphi.
\]
The actual maximum defines \(q_B=\varphi_*/2\). The exact identity
\[
 (u^2+2\Omega_B)'=-4\beta_2'\Omega_B
\]
retains geometric work. On an interval containing the maximum, set
\(\epsilon=\sqrt\delta\) and require uniform bounds
\(c\epsilon\le u_-\le C\epsilon\),
\(0\le\beta'_{2,-}\le C\epsilon^2\),
\(0<v_0\le\sum\beta'_{a,-}\le v_1\), and \(\varphi\ge q_B\).
Both endpoint energies must be small:
\(\eta_B=(\Omega_{B,-}+\Omega_{B,+})/\epsilon^2\ll c^2\).
The outgoing endpoint lies on the decreasing magnetic branch.

The integrated magnetic impulse is \(O(\epsilon/q_B)\).
Scalar reversal and volume normalization give
\[
 d_K(P_+,\mathcal R_\infty P_-)/\epsilon
 \le C(q_B^{-1}+\eta_B)+\nu,
\]
where the reflection reverses the scalar entry and \(\nu\) is normalized
extraction error. The Liouville inner regime has \(q_B\gg1\) and
\(\epsilon q_B\ll1\). Matching both endpoints at
\(\Omega_{B,\pm}/\epsilon^2=c_\pm q_B^{-2}\) gives Liouville tails of
length \(O(\log q_B)\) and a weighted scalar-force remainder
\(O(\epsilon\log q_B/q_B^2)\). The abbreviated \(O(q_B^{-1})\) result
requires endpoint and extraction errors of that order. Additional
compatible sources retain explicit accumulated impulse and work errors.

Eight local controls use measured backgrounds at delta=1e-6 and 1e-8,
initial canonical scalar 10,20,40,80, and
\(\Omega_{B,K}=10^{-7}\delta\).
Logarithmic wall variables avoid overflow and underflow.
DOP853 and exact rescaled Radau integrations locate the peak by a continuous
root and compare both platform windows. Four-dimensional equations,
constraints and finite-difference residuals are checked separately.
