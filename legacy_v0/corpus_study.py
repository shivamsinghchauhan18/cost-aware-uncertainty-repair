"""Preliminary study: how often does the cheapest repair involve a different team
than the smallest repair, and what system structure makes that more likely?

Random systems: 3-6 teams, one private uncertainty source per team plus 0-2
shared sources (owned by a random team) that act on several state variables;
1-3 requirements with random affine mappings; limits set so the system is
currently violated. Team prices are log-normal with dispersion sigma
(sigma = 0 means "all teams cost the same", i.e. the current method).
"""
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from repair import System, price_of_restriction, solve

rng = np.random.default_rng(7)
SIGMAS = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0]
N_PER = 400


def random_system(n_shared: int, sigma: float) -> System:
    n = rng.integers(3, 7)
    teams = [f"T{i}" for i in range(n)]
    G_priv = np.diag(rng.uniform(0.5, 2.0, n))
    G_shared = np.zeros((n, n_shared))
    owners_shared = []
    for k in range(n_shared):
        hit = rng.choice(n, size=rng.integers(2, n + 1), replace=False)
        G_shared[hit, k] = rng.uniform(0.3, 1.2, len(hit))
        owners_shared.append(teams[rng.integers(n)])
    G = np.hstack([G_priv, G_shared])
    r = rng.integers(1, 4)
    F = rng.integers(0, 3, size=(r, n)).astype(float)
    F[F.sum(axis=1) == 0, 0] = 1.0
    sys = System(G=G, F=F, eps=np.zeros(r), owner=teams + owners_shared,
                 s_max=np.full(G.shape[1], 0.8),
                 weight={t: float(np.exp(sigma * rng.standard_normal())) for t in teams})
    sys.eps = sys.spreads() * rng.uniform(0.65, 0.9, r)   # currently violated, but repairable
    return sys


rows = []
for n_shared in (0, 2):
    for sigma in SIGMAS:
        diff_main = diff_set = n = 0
        gaps, ratios, por_inf = [], [], []
        while n < N_PER:
            sys = random_system(n_shared, sigma)
            base, cheap = solve(sys, "uniform"), solve(sys, "unit")
            if not (base.feasible and cheap.feasible):
                continue
            n += 1
            gaps += [base.duality_gap, cheap.duality_gap]
            diff_main += base.main_team() != cheap.main_team()
            diff_set += base.changed_teams() != cheap.changed_teams()
            w = np.array([sys.weight[o] for o in sys.owner]) * np.abs(sys.G).sum(axis=0)
            ratios.append((base.s @ w) / cheap.cost)
            t = cheap.main_team()
            por_inf.append(not np.isfinite(price_of_restriction(sys, "unit", t)))
        rows.append(dict(shared=n_shared, sigma=sigma, main=diff_main / n, set=diff_set / n,
                         overpay=float(np.median(ratios)), infeasible=float(np.mean(por_inf)),
                         maxgap=float(np.max(gaps))))

print(f"{'shared':>6} {'sigma':>5} {'main team differs':>18} {'team set differs':>17} "
      f"{'median overpay':>15} {'no repair w/o main team':>24} {'max dual gap':>13}")
for r in rows:
    print(f"{r['shared']:>6} {r['sigma']:>5} {r['main']:>18.1%} {r['set']:>17.1%} "
          f"{r['overpay']:>14.2f}x {r['infeasible']:>24.1%} {r['maxgap']:>13.1e}")

fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
for n_shared, style in ((0, "o-"), (2, "s--")):
    sub = [r for r in rows if r["shared"] == n_shared]
    label = "independent sources" if n_shared == 0 else "2 shared sources"
    ax[0].plot(SIGMAS, [100 * r["main"] for r in sub], style, label=label)
    ax[1].plot(SIGMAS, [r["overpay"] for r in sub], style, label=label)
ax[0].set(xlabel="cost dispersion between teams (sigma)", ylabel="% systems",
          title="Cheapest repair puts the main burden\non a different team than the smallest repair")
ax[1].set(xlabel="cost dispersion between teams (sigma)", ylabel="median cost ratio",
          title="Cost of the smallest repair /\ncost of the cheapest repair")
for a in ax:
    a.grid(alpha=0.3)
    a.legend()
fig.tight_layout()
fig.savefig("corpus_study.png", dpi=160)
print("\nsaved corpus_study.png")
