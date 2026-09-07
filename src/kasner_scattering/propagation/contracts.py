"""Numerical equations and integration routines for the local EMS model."""
from __future__ import annotations
from dataclasses import asdict, dataclass, field
from math import isfinite, pi, sqrt
from typing import Any

@dataclass(frozen=True)
class PacketSpec:
    delta: float
    omega_times_mass: float = 0.2
    kappa: float = 0.5
    tau0: float = 1.0
    phase: float = 0.0
    envelope: str = 'gaussian'
    input_spectrum: dict[str, Any] | None = None
    energy_normalization: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not all((isfinite(v) for v in (self.delta, self.omega_times_mass, self.kappa, self.tau0, self.phase))):
            raise ValueError('packet parameters must be finite')
        if min(self.delta, self.omega_times_mass, self.kappa, self.tau0) <= 0:
            raise ValueError('packet scales must be positive')
        if self.envelope not in {'gaussian', 'compact'}:
            raise ValueError('unsupported envelope')

    @property
    def amplitude(self) -> float:
        return self.delta ** (0.5 + self.kappa)

    @property
    def duration(self) -> float:
        return self.tau0 / sqrt(self.delta)

    @property
    def gaussian_energy(self) -> float:
        return sqrt(pi) * self.amplitude ** 2 * self.duration

@dataclass
class MatchingData:
    background: dict[str, Any]
    canonical: list
    canonical_momentum: list
    components: dict[str, list]
    polarization: list
    flux_residual: float
    source_hashes: dict[str, str]
    nonlinear_metric_ready: bool = False
    full_horizon_second_order_ready: bool = False
    omissions: list[str] = field(default_factory=list)
    frequency_data: dict[str, Any] | None = None
    linear_horizon_data_ready: bool = False
    local_constraints_ready: bool = False

@dataclass
class EvolutionResult:
    status: str
    model: str
    diagnostics: dict[str, Any]
    p_minus: list[float] | None = None
    p_plus: list[float] | None = None
    scattering_error: float | None = None
    nonlinear_horizon_embedding: bool = False
    prediction: dict[str, Any] | None = None
    validity: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
