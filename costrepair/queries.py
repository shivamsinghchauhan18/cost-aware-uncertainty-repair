"""Higher-level queries: price of restriction, comparison against the current method."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .problem import CostModel, Repair, RepairProblem


def price_of_restriction(problem: RepairProblem, cost: CostModel, teams: set[str],
                         free: Repair | None = None) -> float:
    """Extra cost for everyone when `teams` cannot change. inf if no repair exists without them."""
    free = free or problem.solve(cost)
    if not free.feasible:
        return np.nan
    restricted = problem.solve(cost, frozen=teams)
    return restricted.objective - free.objective if restricted.feasible else np.inf


def por_table(problem: RepairProblem, cost: CostModel) -> dict[str, float]:
    free = problem.solve(cost)
    return {t: price_of_restriction(problem, cost, {t}, free) for t in problem.arch.teams()}


@dataclass
class Comparison:
    baseline: Repair
    costaware: Repair

    @property
    def main_team_differs(self) -> bool:
        return self.baseline.main_team() != self.costaware.main_team()

    @property
    def team_set_differs(self) -> bool:
        return self.baseline.changed_teams() != self.costaware.changed_teams()

    @property
    def overpay(self) -> float:
        """Cost of the current method's repair, priced like the cost-aware one, divided by the optimum."""
        if not (self.baseline.feasible and self.costaware.feasible) or self.costaware.unit_cost <= 1e-12:
            return np.nan
        return self.baseline.unit_cost / self.costaware.unit_cost


def compare(problem: RepairProblem, cost: CostModel, baseline: str = "smallest") -> Comparison:
    base = problem.solve(CostModel(kind=baseline), price_for_report=cost)
    best = problem.solve(cost, price_for_report=cost)
    return Comparison(base, best)
