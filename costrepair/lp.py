"""Tiny LP builder on top of scipy's HiGHS, with a duality-gap certificate."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix

INF = np.inf


@dataclass
class LPSolution:
    status: str                 # "optimal" | "infeasible" | "unbounded" | "error"
    x: np.ndarray | None = None
    objective: float = INF
    dual_objective: float = np.nan
    ineq_duals: np.ndarray | None = None   # d objective / d rhs, one per <= row
    eq_duals: np.ndarray | None = None

    @property
    def gap(self) -> float:
        return abs(self.objective - self.dual_objective)


class LP:
    """min c^T x  s.t.  A_ub x <= b_ub,  A_eq x = b_eq,  lb <= x <= ub."""

    def __init__(self) -> None:
        self.c: list[float] = []
        self.lb: list[float] = []
        self.ub: list[float] = []
        self.names: list[str] = []
        self._ub_rows: list[dict[int, float]] = []
        self._ub_rhs: list[float] = []
        self._eq_rows: list[dict[int, float]] = []
        self._eq_rhs: list[float] = []
        self.row_tags: list[str] = []          # tag per <= row (for shadow prices)

    # -- variables -----------------------------------------------------------
    def var(self, name: str, lb: float = 0.0, ub: float = INF, cost: float = 0.0) -> int:
        self.c.append(cost)
        self.lb.append(lb)
        self.ub.append(ub)
        self.names.append(name)
        return len(self.c) - 1

    def add_cost(self, i: int, cost: float) -> None:
        self.c[i] += cost

    # -- constraints (expressions are {var_index: coeff}) ----------------------
    def le(self, expr: dict[int, float], rhs: float, tag: str = "") -> int:
        self._ub_rows.append(dict(expr))
        self._ub_rhs.append(float(rhs))
        self.row_tags.append(tag)
        return len(self._ub_rhs) - 1

    def eq(self, expr: dict[int, float], rhs: float) -> int:
        self._eq_rows.append(dict(expr))
        self._eq_rhs.append(float(rhs))
        return len(self._eq_rhs) - 1

    @staticmethod
    def _matrix(rows: list[dict[int, float]], n: int):
        if not rows:
            return None
        data, ri, ci = [], [], []
        for r, row in enumerate(rows):
            for j, v in row.items():
                if v != 0.0:
                    data.append(v)
                    ri.append(r)
                    ci.append(j)
        return csr_matrix((data, (ri, ci)), shape=(len(rows), n))

    def solve(self) -> LPSolution:
        n = len(self.c)
        A_ub = self._matrix(self._ub_rows, n)
        A_eq = self._matrix(self._eq_rows, n)
        b_ub = np.array(self._ub_rhs) if self._ub_rows else None
        b_eq = np.array(self._eq_rhs) if self._eq_rows else None
        bounds = list(zip(self.lb, [None if u == INF else u for u in self.ub]))
        bounds = [(None if l == -INF else l, u) for l, u in bounds]
        res = linprog(np.array(self.c), A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                      bounds=bounds, method="highs")
        if res.status == 2:
            return LPSolution("infeasible")
        if res.status == 3:
            return LPSolution("unbounded")
        if res.status != 0:
            return LPSolution("error")
        # Dual objective from HiGHS marginals (sensitivities of the optimum).
        dual = 0.0
        if b_ub is not None:
            dual += b_ub @ res.ineqlin.marginals
        if b_eq is not None:
            dual += b_eq @ res.eqlin.marginals
        lb, ub = np.array(self.lb), np.array(self.ub)
        dual += np.where(np.isfinite(lb), lb, 0.0) @ res.lower.marginals
        dual += np.where(np.isfinite(ub), ub, 0.0) @ res.upper.marginals
        return LPSolution(
            "optimal", x=res.x, objective=float(res.fun), dual_objective=float(dual),
            ineq_duals=res.ineqlin.marginals if b_ub is not None else np.zeros(0),
            eq_duals=res.eqlin.marginals if b_eq is not None else np.zeros(0),
        )
