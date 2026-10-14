"""E2: cost-aware vs. smallest repair on the 12 IRIS-CPS engineering scenarios.

For every scenario we apply the paper's own published Saltelli interventions
(scale, center shift, correlation) and keep the configurations that violate the
requirement. Two certified requirement notions bracket their probabilistic I(theta) <= 0.5:
  overlap      : the mapped source and the target share at least one realization (I < 1)
  containment  : every mapped source realization is admissible in the target   (I = 0)
Teams: one team per source quantity (owns that generator and that center), and
the target team (may shift or widen the target). Prices are drawn per instance
from archetypes (shrinking uncertainty = hardware, shifting = recalibration,
widening the target = relaxing a tolerance), log-normal around the archetype;
with probability 0.15 a team cannot shrink at all.
"""
import csv
import json
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from costrepair import (Architecture, BoxContainment, BoxTarget, CostModel, Model, Overlap,
                        RepairProblem, check, compare, requirement_holds)
from costrepair.queries import price_of_restriction
from iris_scenarios import SCENARIOS, apply_compound

CSV = ROOT / "experiments" / "saltelli_samples_3param_v6.csv"
CSV_URL = "https://raw.githubusercontent.com/DE-TUM/IRIS-CPS/main/data/saltelli_samples_3param_v6.csv"
PER_SCENARIO = 120
rng = np.random.default_rng(2026)


def load_samples():
    if not CSV.exists():
        urllib.request.urlretrieve(CSV_URL, CSV)
    with open(CSV) as fh:
        return [tuple(map(float, r)) for r in list(csv.reader(fh))[1:]]


def build(sc, theta, mode):
    c, G = apply_compound(sc, *theta)
    d = len(c)
    teams = [f"Quantity {i + 1}" for i in range(d)]
    src = Model("Source", c, G, "Source", var_names=[f"q{i + 1}" for i in range(d)],
                gen_teams=teams)
    src.team = teams[0]
    arch = Architecture([src])
    arch.var_team = teams                                # each quantity's center belongs to its team
    target = BoxTarget(sc.tgt_c, sc.tgt_h, "Target", shift_max=float(3 * sc.tgt_h.max()),
                       widen_max=1.0)
    M = sc.F.copy()                                      # mapping x -> F x + f, folded into the target
    tgt = BoxTarget(sc.tgt_c - sc.f, sc.tgt_h, "Target", target.shift_max, target.widen_max)
    req = Overlap("overlap", M, tgt) if mode == "overlap" else BoxContainment("containment", M, tgt)
    gw = np.abs(G).sum(axis=0)
    return RepairProblem(arch, [req], shrink_max=0.9, widen_max=1.0 if mode == "overlap" else 0.0,
                         shift_max=3 * np.abs(G).sum(axis=1).max())


def prices(teams):
    p = {}
    for t in teams:
        if t == "Target":
            p[t] = {"target_widen": 3 * rng.lognormal(0, 1), "target_shift": 3 * rng.lognormal(0, 1)}
        else:
            p[t] = {"shrink": np.inf if rng.random() < 0.15 else 5 * rng.lognormal(0, 1),
                    "widen": 1 * rng.lognormal(0, 1), "shift": 2 * rng.lognormal(0, 1)}
    return p


def main():
    samples = load_samples()
    order = rng.permutation(len(samples))
    rows = []
    t0 = time.time()
    for mode in ("overlap", "containment"):
        for sc in SCENARIOS:
            taken = 0
            for idx in order:
                if taken >= PER_SCENARIO:
                    break
                prob = build(sc, samples[idx], mode)
                if requirement_holds(prob):
                    continue                                  # only repair inconsistent configurations
                cost = CostModel("unit", prices(prob.arch.teams()))
                cmp = compare(prob, cost)
                if not cmp.costaware.feasible:
                    rows.append(dict(mode=mode, sid=sc.sid, d=len(sc.src_c), feasible=False))
                    taken += 1
                    continue
                vb, vc = check(prob, cmp.baseline), check(prob, cmp.costaware)
                main = cmp.costaware.main_team()
                por = price_of_restriction(prob, cost, {main}, cmp.costaware) if main else 0.0
                rows.append(dict(mode=mode, sid=sc.sid, d=len(sc.src_c), feasible=True,
                                 main_differs=cmp.main_team_differs, set_differs=cmp.team_set_differs,
                                 overpay=cmp.overpay, verified=vb["ok"] and vc["ok"],
                                 gap=max(cmp.baseline.gap, cmp.costaware.gap),
                                 por_inf=not np.isfinite(por), base_main=cmp.baseline.main_team(),
                                 cost_main=main, theta=samples[idx]))
                taken += 1
    dt = time.time() - t0

    print(f"{len(rows)} inconsistent configurations repaired in {dt:.0f}s "
          f"({1000 * dt / max(len(rows), 1) / 4:.1f} ms per LP incl. verification)\n")
    print(f"{'mode':<12}{'sid':>4} {'name':<32}{'n':>4} {'main differs':>13} {'set differs':>12} "
          f"{'med overpay':>12} {'p90 overpay':>12} {'SPOF':>6} {'verified':>9}")
    summary = defaultdict(dict)
    for mode in ("overlap", "containment"):
        for sc in SCENARIOS + [None]:
            sub = [r for r in rows if r["mode"] == mode and r["feasible"] and (sc is None or r["sid"] == sc.sid)]
            if not sub:
                continue
            ov = np.array([r["overpay"] for r in sub if np.isfinite(r["overpay"])])
            s = dict(n=len(sub), main=np.mean([r["main_differs"] for r in sub]),
                     set=np.mean([r["set_differs"] for r in sub]),
                     med=float(np.median(ov)) if len(ov) else np.nan,
                     p90=float(np.quantile(ov, 0.9)) if len(ov) else np.nan,
                     spof=np.mean([r["por_inf"] for r in sub]),
                     ver=np.mean([r["verified"] for r in sub]), gap=max(r["gap"] for r in sub))
            key = "ALL" if sc is None else sc.sid
            summary[mode][key] = s
            name = "ALL SCENARIOS" if sc is None else sc.name
            print(f"{mode:<12}{str(key):>4} {name:<32}{s['n']:>4} {s['main']:>13.0%} {s['set']:>12.0%} "
                  f"{s['med']:>11.2f}x {s['p90']:>11.2f}x {s['spof']:>6.0%} {s['ver']:>9.0%}")
        print()
    infeas = sum(not r["feasible"] for r in rows)
    print(f"no repair possible within lever bounds: {infeas} configurations")
    print(f"max LP duality gap over all repairs: {max(r['gap'] for r in rows if r['feasible']):.1e}")
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "e2_iris_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    (ROOT / "results" / "e2_iris_rows.json").write_text(json.dumps(rows, default=float))


if __name__ == "__main__":
    main()
