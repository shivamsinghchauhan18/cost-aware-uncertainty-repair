"""Exact Python port of the 12 IRIS-CPS engineering scenarios and their compound interventions.

Source: github.com/DE-TUM/IRIS-CPS, examples/generate_engineering_twostep.m (scenario
definitions and affine uncertainty mappings) and src/matlab/causal_experiment_engine_twostep.m
(apply_compound_intervention: scale -> shift -> correlate). Saltelli samples are the
published data/saltelli_samples_3param_v6.csv.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _rot2(a):
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


_a = 5 * np.pi / 180
_b = 3 * np.pi / 180
R2 = _rot2(_a)
R3 = np.eye(3)
R3[:2, :2] = _rot2(_a)
R4 = np.zeros((4, 4))
R4[:2, :2] = _rot2(_a)
R4[2:, 2:] = _rot2(_b)


@dataclass
class IrisScenario:
    sid: int
    name: str
    src_c: np.ndarray
    src_G: np.ndarray
    tgt_c: np.ndarray
    tgt_h: np.ndarray        # target box half-widths (targets are diagonal conZonotopes)
    F: np.ndarray
    f: np.ndarray


def _s(sid, name, sc, sg, tc, tg, F, f=None):
    sc, tc = np.array(sc, float), np.array(tc, float)
    return IrisScenario(sid, name, sc, np.diag(sg).astype(float), tc, np.array(tg, float),
                        np.array(F, float), np.zeros(len(sc)) if f is None else np.array(f, float))


SCENARIOS = [
    _s(1, "CAD Export Drift", [100, 50], [2.0, 2.0], [100.3, 50.3], [2.1, 2.1], np.eye(2)),
    _s(2, "MBSE Version Mismatch", [50, 25], [1.5, 1.5], [50, 25], [1.8, 1.8], np.diag([0.97, 1.04])),
    _s(3, "Documentation Sync", [75, 40], [2.0, 2.0], [75.5, 40.5], [2.0, 2.0], np.eye(2), [0.5, 0.3]),
    _s(4, "Control Design Conflict", [60, 30], [1.8, 1.8], [60.8, 30.8], [1.5, 1.5], R2),
    _s(5, "Sensor Calibration Drift", [100, 50, 25], [2.5, 2.5, 1.2], [100.5, 50.5, 25.5], [2.6, 2.6, 1.3], np.eye(3)),
    _s(6, "Requirements Ambiguity", [80, 40, 20], [2.0, 2.0, 1.0], [80, 40, 20], [2.5, 2.5, 1.2], np.diag([0.95, 1.05, 0.98])),
    _s(7, "Test Configuration Mismatch", [70, 35, 18], [2.2, 2.2, 1.1], [70.4, 35.4, 18.4], [2.2, 2.2, 1.1], np.eye(3), [0.3, -0.2, 0.1]),
    _s(8, "Simulation Numerical Error", [90, 45, 22], [2.5, 2.5, 1.3], [90.6, 45.6, 22.6], [2.2, 2.2, 1.1], R3),
    _s(9, "Multi-Physics Coupling Error", [100, 50, 25, 12], [3.0, 3.0, 1.5, 0.75], [100.5, 50.5, 25.5, 12.5], [3.1, 3.1, 1.6, 0.8], np.eye(4)),
    _s(10, "Interface Specification Gap", [85, 42, 21, 10], [2.5, 2.5, 1.2, 0.6], [85, 42, 21, 10], [3.0, 3.0, 1.5, 0.75], np.diag([0.96, 1.03, 0.99, 1.02])),
    _s(11, "Parameter Estimation Bias", [95, 48, 24, 11], [2.8, 2.8, 1.4, 0.7], [95.4, 48.4, 24.4, 11.4], [2.8, 2.8, 1.4, 0.7], np.eye(4), [0.4, -0.3, 0.2, -0.1]),
    _s(12, "Traceability Link Inconsistency", [75, 38, 19, 9], [2.5, 2.5, 1.2, 0.6], [75.6, 38.6, 19.6, 9.6], [2.2, 2.2, 1.1, 0.55], R4),
]


def apply_compound(sc: IrisScenario, scale: float, delta: float, rho: float) -> tuple[np.ndarray, np.ndarray]:
    """Their apply_compound_intervention: scale -> shift (delta generator widths) -> correlate."""
    G = scale * sc.src_G
    c = sc.src_c + delta * np.diag(G)
    if G.shape[1] >= 2:
        G = G.copy()
        G[:, 1] = G[:, 0] * rho + G[:, 1] * np.sqrt(1 - rho ** 2)
    return c, G
