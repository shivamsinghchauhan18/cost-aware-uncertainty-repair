"""E3: which system structures make the cost model matter?

Random multi-model architectures with controlled structure:
  topology      chain | star (hub feeds all) | sink (all feed one)
  n_models      2..5 (one team each) + optional shared Infra source feeding k models
  constrained   models may carry a constraint row (constrained zonotope)
  requirements  1..3 variability requirements on random combinations of model outputs
Per instance we record structural features and outcomes, then fit a standardized
logistic regression (main team differs ~ features) and report odds ratios with
bootstrap confidence intervals, plus binned effect plots.
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from costrepair import (Architecture, CostModel, Mapping, Model, RepairProblem, Variability,
                        check, compare, requirement_holds)
from costrepair.queries import price_of_restriction

rng = np.random.default_rng(11)
N = 2500


def random_instance():
    topo = rng.choice(["chain", "star", "sink"])
    n = int(rng.integers(2, 6))
    shared = int(rng.integers(0, n + 1))                  # how many models the Infra source feeds
    constrained = rng.random() < 0.5
    sigma_cost = float(rng.choice([0.5, 1.0, 1.5]))
    models, maps = [], []
    if shared:
        models.append(Model("Infra", [0.0], [[rng.uniform(0.3, 1.2)]], "Infra"))
    for i in range(n):
        g = int(rng.integers(1, 3))
        G = rng.uniform(0.3, 2.0, (1, g))
        A = b = None
        if constrained and g == 2:
            G = np.hstack([G, [[0.0]]])
            A = np.array([[1.0, 1.0, rng.uniform(0.3, 1.0)]])
            b = np.array([0.0])
        models.append(Model(f"M{i}", [rng.uniform(10, 50)], G, f"Team{i}", A=A, b=b))
    for i in range(n):
        if topo == "chain" and i > 0:
            maps.append(Mapping(f"M{i - 1}", f"M{i}", [[rng.uniform(0.5, 1.5)]]))
        if topo == "star" and i > 0:
            maps.append(Mapping("M0", f"M{i}", [[rng.uniform(0.5, 1.5)]]))
        if topo == "sink" and i < n - 1:
            maps.append(Mapping(f"M{i}", f"M{n - 1}", [[rng.uniform(0.5, 1.5)]]))
    for j in rng.choice(n, size=shared, replace=False) if shared else []:
        maps.append(Mapping("Infra", f"M{j}", [[1.0]]))
    arch = Architecture(models, maps)
    model_vars = [arch.var(f"M{i}.x0") for i in range(n)]
    reqs = []
    for r in range(int(rng.integers(1, 4))):
        w = np.zeros(arch.n)
        pick = rng.choice(model_vars, size=int(rng.integers(1, n + 1)), replace=False)
        w[pick] = rng.integers(1, 3, len(pick))
        reqs.append(Variability(f"R{r}", w, 1.0, "Product"))
    prob = RepairProblem(arch, reqs, shrink_max=0.85)
    for req in reqs:                                       # set limits so each is violated
        cur = (np.abs(arch.G.T @ req.w).sum())             # (upper bound on the spread)
        req.eps = cur * rng.uniform(0.55, 0.85)
    teams = arch.teams()
    prices = {t: {"shrink": float(np.exp(sigma_cost * rng.standard_normal()))} for t in teams}
    return prob, CostModel("unit", prices), dict(topology=topo, n_models=n, shared=shared,
                                                 constrained=constrained, sigma_cost=sigma_cost,
                                                 n_req=len(reqs))


def features(prob, cost, meta):
    """Structure features of the instance (computed before repair)."""
    a = prob.arch
    # per-team contribution to the violated spreads (sensitivity), and their balance
    contrib = {}
    for req in prob.requirements:
        av = np.abs(a.G.T @ req.w)
        for k, v in enumerate(av):
            contrib[a.gen_team[k]] = contrib.get(a.gen_team[k], 0.0) + v
    teams = [t for t in contrib if contrib[t] > 0]
    share = np.array([contrib[t] for t in teams]) / sum(contrib[t] for t in teams)
    entropy = float(-(share * np.log(share)).sum() / np.log(len(share))) if len(share) > 1 else 0.0
    price = np.array([cost.price(t, "shrink") for t in teams])
    align = spearmanr(share, price).statistic if len(teams) > 2 else 0.0
    gens_per_team = np.mean([sum(1 for t_ in a.gen_team if t_ == t) for t in teams])
    return dict(meta, n_teams=len(teams), balance=entropy, price_sensitivity_corr=float(np.nan_to_num(align)),
                gens_per_team=float(gens_per_team))


def logistic(X, y):
    def nll(beta):
        z = X @ beta
        return np.sum(np.logaddexp(0, z) - y * z) + 1e-3 * beta[1:] @ beta[1:]
    return minimize(nll, np.zeros(X.shape[1]), method="BFGS").x


def main():
    rows = []
    while len(rows) < N:
        prob, cost, meta = random_instance()
        if requirement_holds(prob):
            continue
        cmp = compare(prob, cost)
        if not cmp.costaware.feasible:
            continue
        assert check(prob, cmp.costaware, mc_samples=0)["ok"]
        f = features(prob, cost, meta)
        main = cmp.costaware.main_team()
        por = price_of_restriction(prob, cost, {main}, cmp.costaware) if main else 0.0
        rows.append(dict(f, main_differs=cmp.main_team_differs, overpay=cmp.overpay,
                         spof=not np.isfinite(por)))

    names = ["n_teams", "shared", "constrained", "sigma_cost", "n_req", "balance",
             "price_sensitivity_corr", "gens_per_team", "topo_star", "topo_sink"]
    X = np.array([[r["n_teams"], r["shared"], float(r["constrained"]), r["sigma_cost"], r["n_req"],
                   r["balance"], r["price_sensitivity_corr"], r["gens_per_team"],
                   float(r["topology"] == "star"), float(r["topology"] == "sink")] for r in rows])
    mu, sd = X.mean(0), X.std(0)
    Xs = np.hstack([np.ones((len(X), 1)), (X - mu) / np.where(sd > 0, sd, 1)])
    y = np.array([float(r["main_differs"]) for r in rows])
    beta = logistic(Xs, y)
    boots = []
    for _ in range(200):
        idx = rng.integers(0, len(y), len(y))
        boots.append(logistic(Xs[idx], y[idx]))
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)

    print(f"{len(rows)} repaired instances; main team differs in {y.mean():.0%}; "
          f"median overpay {np.nanmedian([r['overpay'] for r in rows]):.2f}x; "
          f"single point of failure in {np.mean([r['spof'] for r in rows]):.0%}\n")
    print("Logistic regression: P(cheapest repair picks a different main team), standardized features")
    print(f"{'feature':<24}{'odds ratio / +1 SD':>20}{'95% bootstrap CI':>24}")
    effects = {}
    for i, nm in enumerate(names, start=1):
        sig = "*" if (lo[i] > 0 or hi[i] < 0) else " "
        print(f"{nm:<24}{np.exp(beta[i]):>20.2f}   [{np.exp(lo[i]):.2f}, {np.exp(hi[i]):.2f}] {sig}")
        effects[nm] = dict(odds_ratio=float(np.exp(beta[i])), ci=[float(np.exp(lo[i])), float(np.exp(hi[i]))])

    # binned effect plots for the strongest structural drivers
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    def binned(key, bins, axis, label):
        vals = np.array([r[key] for r in rows], float)
        idx = np.digitize(vals, bins)
        xs, ys, es = [], [], []
        for b in np.unique(idx):
            sel = idx == b
            if sel.sum() < 30:
                continue
            p = y[sel].mean()
            xs.append(vals[sel].mean()); ys.append(100 * p); es.append(100 * 1.96 * np.sqrt(p * (1 - p) / sel.sum()))
        axis.errorbar(xs, ys, yerr=es, fmt="o-", capsize=3)
        axis.set(xlabel=label, ylabel="% main team differs")
        axis.grid(alpha=0.3)
    binned("balance", np.linspace(0, 1, 6), ax[0], "balance of team contributions (0 = one team dominates)")
    binned("price_sensitivity_corr", np.linspace(-1, 1, 6), ax[1], "corr(team contribution, team price)")
    binned("n_teams", np.arange(1.5, 8), ax[2], "teams contributing to the violation")
    fig.suptitle("E3: when does cost change who should fix it?  (n = %d random architectures)" % len(rows))
    fig.tight_layout()
    fig.savefig(ROOT / "results" / "e3_structure.png", dpi=160)
    (ROOT / "results" / "e3_structure.json").write_text(json.dumps(
        dict(n=len(rows), base_rate=float(y.mean()), effects=effects), indent=2))
    print("\nsaved results/e3_structure.png")


if __name__ == "__main__":
    main()
