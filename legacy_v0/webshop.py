"""The web-shop example from the thesis topic: who should fix checkout latency?

State x = [database, auth, gateway] service times (ms).
Checkout calls the database twice, then auth and the gateway; login calls auth
and the gateway. A shared network-jitter source (owned by the infrastructure
provider) affects both auth and gateway calls, so the generators are not all
axis-aligned.
"""
import numpy as np

from repair import System, price_of_restriction, solve

G = np.array([
    #  db    auth  gw    net-jitter
    [1.5,  0.0,  0.0,  0.0],   # database
    [0.0,  1.2,  0.0,  0.6],   # auth
    [0.0,  0.0,  1.0,  0.6],   # gateway
])
F = np.array([
    [2, 1, 1],   # checkout latency = 2*db + auth + gw
    [0, 1, 1],   # login latency    = auth + gw
])
system = System(
    G=G,
    F=F,
    eps=np.array([5.0, 3.0]),            # "checkout may vary by at most +/-5 ms", login +/-3 ms
    owner=["Database", "Auth", "Gateway", "Infra"],
    s_max=np.array([0.8, 0.8, 0.8, 0.5]),  # nobody can remove all uncertainty
    # EUR per ms of uncertainty removed (unit model) / per 100 % reduction (proportional):
    # Database needs new hardware; Auth is a config tweak; Gateway moderate; Infra = contract.
    weight={"Database": 8000, "Auth": 500, "Gateway": 2000, "Infra": 6000},
)


def show(title, rep, sys=system):
    print(f"\n== {title}")
    if not rep.feasible:
        print("   no repair exists")
        return
    for t in sys.teams():
        print(f"   {t:<9} removes {rep.effort_by_team[t]:5.2f} ms of uncertainty")
    print(f"   spreads after repair: {np.round(sys.spreads(rep.s), 3)}  (limits {sys.eps})")
    print(f"   objective {rep.cost:,.2f}   dual {rep.dual_objective:,.2f}   gap {rep.duality_gap:.1e}  -> certified optimal")
    print(f"   shadow prices (per ms of tighter requirement): {np.round(rep.shadow_prices, 2)}")


if __name__ == "__main__":
    print("current spreads (checkout, login):", system.spreads(), " limits:", system.eps)

    base = solve(system, "uniform")
    show("Current method: smallest revision, every team's change costs the same", base)
    unit = solve(system, "unit")
    show("Cost-aware (unit costs, EUR/ms)", unit)
    prop = solve(system, "proportional")
    show("Cost-aware (proportional costs)", prop)

    eur = lambda r: (r.s @ (np.array([system.weight[o] for o in system.owner]) * np.abs(G).sum(axis=0))) if r.feasible else float("inf")
    print(f"\nEUR cost of the smallest revision: {eur(base):,.0f}  vs cheapest repair: {unit.cost:,.0f}"
          f"  -> the 'smallest' fix is {eur(base) / unit.cost:.1f}x more expensive")
    print(f"main team, smallest revision: {base.main_team()} | cheapest repair: {unit.main_team()}")

    print("\nPrice of restriction (unit costs): extra EUR when one team cannot change")
    for t in system.teams():
        p = price_of_restriction(system, "unit", t)
        print(f"   {t:<9} frozen -> +{p:,.0f} EUR" if np.isfinite(p) else f"   {t:<9} frozen -> no repair possible")
