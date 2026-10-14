"""Independent verification of a repair.

The solver certifies requirements through *dual* support-function bounds. Here we
rebuild the repaired sets and re-check every requirement with *primal* LPs
(maximise over xi directly), plus Monte Carlo sampling where the set has no
equality constraints. A repair passes if every check holds within tolerance.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linprog

from .problem import Bound, BoxContainment, Overlap, Repair, RepairProblem, Variability


def _support(a_vec: np.ndarray, A: np.ndarray, b: np.ndarray) -> float:
    """max a^T xi  s.t.  A xi = b, -1 <= xi <= 1   (primal)."""
    m = len(a_vec)
    if np.all(a_vec == 0):
        return 0.0
    if A.shape[0] == 0:
        return float(np.abs(a_vec).sum())
    res = linprog(-a_vec, A_eq=A, b_eq=b, bounds=[(-1, 1)] * m, method="highs")
    if res.status != 0:
        raise RuntimeError("nominal constrained zonotope is empty")
    return float(-res.fun)


def check(problem: RepairProblem, rep: Repair, tol: float = 1e-6, mc_samples: int = 2000,
          rng: np.random.Generator | None = None) -> dict:
    a = problem.arch
    c = a.c + a.T @ rep.dc
    G = a.G * rep.sigma
    worst, mc_viol = -np.inf, 0
    rng = rng or np.random.default_rng(0)
    xi = None
    if a.A.shape[0] == 0 and mc_samples:
        xi = rng.uniform(-1, 1, size=(mc_samples, a.m))
        X = c + xi @ G.T

    for req in problem.requirements:
        if isinstance(req, Bound):
            lhs = req.w @ c + _support(G.T @ req.w, a.A, a.b)
            worst = max(worst, lhs - (req.beta + rep.relax.get(req.name, 0.0)))
            if xi is not None:
                mc_viol += int(np.sum(X @ req.w > req.beta + rep.relax.get(req.name, 0.0) + tol))
        elif isinstance(req, Variability):
            width = _support(G.T @ req.w, a.A, a.b) + _support(-(G.T @ req.w), a.A, a.b)
            worst = max(worst, width / 2 - (req.eps + rep.relax.get(req.name, 0.0)))
        elif isinstance(req, BoxContainment):
            dct, tau = rep.targets[req.name]
            ct, ht = req.target.ct + dct, req.target.ht * tau
            for i in range(req.M.shape[0]):
                for s in (1.0, -1.0):
                    w = s * req.M[i]
                    lhs = w @ c + _support(G.T @ w, a.A, a.b)
                    worst = max(worst, lhs - (s * ct[i] + ht[i]))
            if xi is not None:
                Y = X @ req.M.T
                mc_viol += int(np.sum(np.abs(Y - ct) > ht + tol))
        elif isinstance(req, Overlap):
            # find a witness xi with M x(xi) inside the target box: LP feasibility
            dct, tau = rep.targets[req.name]
            ct, ht = req.target.ct + dct, req.target.ht * tau
            MG, Mc = req.M @ G, req.M @ c
            A_ub = np.vstack([MG, -MG])
            b_ub = np.concatenate([ct + ht - Mc, -(ct - ht) + Mc]) + tol
            res = linprog(np.zeros(a.m), A_ub=A_ub, b_ub=b_ub,
                          A_eq=a.A if a.A.shape[0] else None, b_eq=a.b if a.A.shape[0] else None,
                          bounds=[(-1, 1)] * a.m, method="highs")
            worst = max(worst, 0.0 if res.status == 0 else np.inf)
    return {"ok": bool(worst <= tol and mc_viol == 0), "max_violation": float(worst), "mc_violations": mc_viol}


def requirement_holds(problem: RepairProblem, tol: float = 1e-6) -> bool:
    """Is the *unrepaired* system already compliant?"""
    a = problem.arch
    nominal = Repair(status="optimal", sigma=np.ones(a.m), dc=np.zeros(a.n))
    for req in problem.requirements:
        if isinstance(req, (BoxContainment, Overlap)):
            nominal.targets[req.name] = (np.zeros(req.M.shape[0]), np.ones(req.M.shape[0]))
    return check(problem, nominal, tol, mc_samples=0)["ok"]
