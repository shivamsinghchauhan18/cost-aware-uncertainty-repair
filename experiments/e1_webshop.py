"""E1: the topic's web-shop example as a multi-model architecture.

Models (each owned by a team):
  Infra     network jitter source                                  (Infra, external provider)
  Database  query service time                                     (Database team)
  Auth      token check + session lookup, constrained:             (Auth team)
            the two cannot both be at their worst case (xi_t + xi_s + 0.5 xi_slack = 0)
  Gateway   routing/TLS + jitter from Infra (mapping Infra -> Gateway)
  Checkout  = 2*Database + Auth + Gateway        (mapping, no own uncertainty)
  Login     = Auth + Gateway
Requirements (owned by Product, who may relax them at a price):
  checkout latency may vary by at most +/-5 ms, login by at most +/-2.5 ms,
  and checkout must stay below 128 ms in the worst case.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from costrepair import (Architecture, Bound, CostModel, Mapping, Model, RepairProblem, Variability,
                        check, compare, por_table)

OUT = Path(__file__).resolve().parents[1] / "results"


def build():
    infra = Model("Infra", [0.0], [[0.6]], "Infra", var_names=["jitter"], gen_names=["network jitter"])
    db = Model("Database", [40.0], [[1.5]], "Database", var_names=["db"], gen_names=["query time"])
    auth = Model("Auth", [25.0], [[1.0, 0.7, 0.0]], "Auth", var_names=["auth"],
                 gen_names=["token check", "session lookup", "slack"],
                 A=np.array([[1.0, 1.0, 0.5]]), b=np.array([0.0]))
    gw = Model("Gateway", [20.0], [[1.0]], "Gateway", var_names=["gw"], gen_names=["routing/TLS"])
    checkout = Model("Checkout", [0.0], np.zeros((1, 0)), "Product", var_names=["checkout"])
    login = Model("Login", [0.0], np.zeros((1, 0)), "Product", var_names=["login"])
    maps = [Mapping("Infra", "Auth", [[1.0]]), Mapping("Infra", "Gateway", [[1.0]]),
            Mapping("Database", "Checkout", [[2.0]]), Mapping("Auth", "Checkout", [[1.0]]),
            Mapping("Gateway", "Checkout", [[1.0]]), Mapping("Auth", "Login", [[1.0]]),
            Mapping("Gateway", "Login", [[1.0]])]
    arch = Architecture([infra, db, auth, gw, checkout, login], maps)
    ck, lg = arch.selector("checkout")[0], arch.selector("login")[0]
    reqs = [Variability("checkout varies <= 5 ms", ck, 5.0, "Product", relax_max=2.0),
            Variability("login varies <= 2.5 ms", lg, 2.5, "Product", relax_max=2.0),
            Bound("checkout <= 128 ms", ck, 128.0, "Product", relax_max=10.0, scale=5.0)]
    shrink = np.array([0.5 if t == "Infra" else 0.8 for t in arch.gen_team])
    prob = RepairProblem(arch, reqs, shrink_max=shrink, shift_max=3.0,
                         shiftable_vars={arch.var("db"), arch.var("auth"), arch.var("gw")})
    return arch, prob


# EUR per unit: shrinking uncertainty per ms, shifting a mean per ms, relaxing a requirement per ms
PRICES = {
    "Database": {"shrink": 8000, "shift": 3000},      # needs new hardware
    "Auth":     {"shrink": 500,  "shift": 400},       # configuration tweak
    "Gateway":  {"shrink": 2000, "shift": 1500},
    "Infra":    {"shrink": 6000, "shift": np.inf},    # external contract; cannot move the mean
    "Product":  {"relax": 5000},                      # weaker SLA = lost revenue
}


def describe(name, rep):
    print(f"\n== {name}")
    if not rep.feasible:
        print("   no repair exists")
        return
    for team in sorted(rep.team_unit_cost):
        if rep.team_rel.get(team, 0) > 1e-9:
            print(f"   {team:<9} changes {rep.team_abs[team]:6.3f} units   (EUR {rep.team_unit_cost[team]:>9,.0f})")
    print(f"   total EUR {rep.unit_cost:,.0f}   | LP duality gap {rep.gap:.1e}")
    sp = {k: round(v, 1) for k, v in rep.shadow.items() if abs(v) > 1e-9}
    if sp:
        print(f"   shadow prices (EUR per unit tighter requirement): {sp}")


def main():
    arch, prob = build()
    unit = CostModel("unit", PRICES)
    results = {}
    cmp = compare(prob, unit)
    describe("Current method: smallest revision (every team equal)", cmp.baseline)
    describe("Cost-aware: unit costs", cmp.costaware)
    prop = prob.solve(CostModel("proportional", PRICES), price_for_report=unit)
    describe("Cost-aware: proportional costs", prop)
    pwl = prob.solve(CostModel("pwl", PRICES, pwl=[(0, 1.0), (500, 3.0)]), price_for_report=unit)
    describe("Cost-aware: convex piecewise-linear (spend above EUR 500 per team costs 3x: overtime)", pwl)

    for nm, rep in [("baseline", cmp.baseline), ("unit", cmp.costaware), ("proportional", prop), ("pwl", pwl)]:
        v = check(prob, rep)
        assert v["ok"], (nm, v)
        results[nm] = {"eur": rep.unit_cost, "main_team": rep.main_team(), "verified": v,
                       "gap": rep.gap, "levers": [(t, k, l, round(x, 4)) for t, k, l, x in rep.lever_values]}

    print(f"\nThe smallest revision costs EUR {cmp.baseline.unit_cost:,.0f}; the cheapest repair costs "
          f"EUR {cmp.costaware.unit_cost:,.0f} ({cmp.overpay:.1f}x). Main team: "
          f"{cmp.baseline.main_team()} -> {cmp.costaware.main_team()}")
    print("\nPrice of restriction (unit costs):")
    por = por_table(prob, unit)
    for t, p in por.items():
        print(f"   {t:<9} cannot change -> " + (f"+EUR {p:,.0f}" if np.isfinite(p) else "no repair possible"))
    results["price_of_restriction"] = {k: (v if np.isfinite(v) else "infeasible") for k, v in por.items()}
    print("\nAll four repairs independently verified (primal LP re-check + Monte Carlo where applicable).")
    OUT.mkdir(exist_ok=True)
    (OUT / "e1_webshop.json").write_text(json.dumps(results, indent=2, default=float))


if __name__ == "__main__":
    main()
