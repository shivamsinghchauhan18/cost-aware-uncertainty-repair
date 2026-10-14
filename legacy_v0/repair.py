"""Cost-aware minimal repair of uncertainty requirements, as a linear program.

Model
-----
System state x in R^n is uncertain; its admissible set is a zonotope
    Z = { c + G xi : xi in [-1, 1]^m },   G = [g_1 ... g_m]
where each generator g_k is one uncertainty source, owned by one team.

A requirement j looks at an output y_j = f_j^T x (an affine uncertainty
mapping) and bounds its variability: the half-width of y_j's range must be
at most eps_j. For a zonotope the range of f_j^T x is exact:
    h_j = sum_k |f_j^T g_k| =: sum_k a_jk

A revision shrinks generator k by a fraction s_k in [0, s_max_k]:
    g_k -> (1 - s_k) g_k,   so   h_j(s) = sum_k a_jk (1 - s_k)  (linear in s)

Repair = choose s so that every requirement holds, at minimum cost:
    min  sum_k cost_k * s_k
    s.t. sum_k a_jk (1 - s_k) <= eps_j      for every requirement j
         0 <= s_k <= s_max_k                (0 for sources that cannot change)

Cost models (all keep the problem an LP):
    "uniform"      cost_k = ||g_k||_1          every ms of uncertainty removed costs the same
                                               (the baseline "smallest revision")
    "unit"         cost_k = w_k * ||g_k||_1    team-specific price per ms removed
    "proportional" cost_k = w_k                price per relative change of a source

Certificate: strong LP duality. The dual objective equals the primal
objective at optimum, and the dual variables of the requirement rows are
shadow prices (cost of tightening requirement j by one unit).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linprog


@dataclass
class System:
    G: np.ndarray            # n x m generator matrix (columns = uncertainty sources)
    F: np.ndarray            # r x n requirement mappings (rows = outputs)
    eps: np.ndarray          # r   variability bounds (half-widths)
    owner: list[str]         # m   team owning each generator
    s_max: np.ndarray        # m   max achievable relative reduction per source
    weight: dict[str, float] = field(default_factory=dict)  # price per team

    @property
    def A(self) -> np.ndarray:
        """a_jk = |f_j^T g_k|: contribution of source k to requirement j's spread."""
        return np.abs(self.F @ self.G)

    def spreads(self, s: np.ndarray | None = None) -> np.ndarray:
        s = np.zeros(self.G.shape[1]) if s is None else s
        return self.A @ (1.0 - s)

    def teams(self) -> list[str]:
        return sorted(set(self.owner))


@dataclass
class Repair:
    feasible: bool
    s: np.ndarray | None = None
    cost: float = float("inf")
    dual_objective: float = float("nan")
    duality_gap: float = float("nan")
    shadow_prices: np.ndarray | None = None   # per requirement, >= 0
    effort_by_team: dict[str, float] = field(default_factory=dict)  # ms removed

    def changed_teams(self, tol: float = 1e-7) -> set[str]:
        return {t for t, v in self.effort_by_team.items() if v > tol}

    def main_team(self) -> str | None:
        if not self.feasible or not self.effort_by_team:
            return None
        return max(self.effort_by_team, key=self.effort_by_team.get)


def cost_vector(sys: System, model: str) -> np.ndarray:
    size = np.abs(sys.G).sum(axis=0)                 # ||g_k||_1
    w = np.array([sys.weight.get(o, 1.0) for o in sys.owner])
    if model == "uniform":
        return size
    if model == "unit":
        return w * size
    if model == "proportional":
        return w
    raise ValueError(f"unknown cost model {model!r}")


def solve(sys: System, model: str = "uniform", frozen: set[str] = frozenset()) -> Repair:
    A = sys.A
    c = cost_vector(sys, model)
    ub = np.where([o in frozen for o in sys.owner], 0.0, sys.s_max)
    # sum_k a_jk (1 - s_k) <= eps_j   <=>   -A s <= eps - A 1
    A_ub = -A
    b_ub = sys.eps - A.sum(axis=1)
    bounds = list(zip(np.zeros_like(ub), ub))
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if res.status == 2:                               # infeasible
        return Repair(feasible=False)
    if res.status != 0:
        raise RuntimeError(res.message)

    s = res.x
    # Dual objective (HiGHS marginals are d(objective)/d(rhs)):
    lo = np.array([b[0] for b in bounds])
    dual = b_ub @ res.ineqlin.marginals + lo @ res.lower.marginals + ub @ res.upper.marginals
    size = np.abs(sys.G).sum(axis=0)
    effort: dict[str, float] = {t: 0.0 for t in sys.teams()}
    for k, o in enumerate(sys.owner):
        effort[o] += s[k] * size[k]
    return Repair(
        feasible=True,
        s=s,
        cost=float(res.fun),
        dual_objective=float(dual),
        duality_gap=abs(float(res.fun) - float(dual)),
        shadow_prices=-res.ineqlin.marginals,
        effort_by_team=effort,
    )


def price_of_restriction(sys: System, model: str, team: str) -> float:
    """Extra cost everyone else pays when `team` cannot change (inf if no repair exists)."""
    free = solve(sys, model)
    restricted = solve(sys, model, frozen={team})
    if not free.feasible:
        return float("nan")
    return restricted.cost - free.cost if restricted.feasible else float("inf")
