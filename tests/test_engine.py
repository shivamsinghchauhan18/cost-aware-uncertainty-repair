import numpy as np
import pytest

from costrepair import (Architecture, Bound, BoxContainment, BoxTarget, CostModel, Mapping, Model,
                        Overlap, RepairProblem, Variability, check, por_table, requirement_holds)
from costrepair.verify import _support


def random_conzono_arch(rng, n=3, g=5, q=2):
    G = rng.uniform(-1, 1, (n, g))
    A = rng.uniform(-1, 1, (q, g))
    xi0 = rng.uniform(-0.5, 0.5, g)                 # guarantees a non-empty set
    return Architecture([Model("M", rng.uniform(-1, 1, n), G, "T", A=A, b=A @ xi0,
                               gen_teams=[f"T{k % 3}" for k in range(g)])])


@pytest.mark.parametrize("seed", range(15))
def test_constrained_repair_is_certified_and_tight(seed):
    rng = np.random.default_rng(seed)
    arch = random_conzono_arch(rng)
    w = rng.uniform(-1, 1, arch.n)
    nominal_max = w @ arch.c + _support(arch.G.T @ w, arch.A, arch.b)
    nominal_min = w @ arch.c - _support(-(arch.G.T @ w), arch.A, arch.b)
    beta = nominal_min + 0.6 * (nominal_max - nominal_min)
    prob = RepairProblem(arch, [Bound("b", w, beta, "Req")], shrink_max=0.95)
    rep = prob.solve(CostModel("smallest"))
    assert rep.feasible
    assert rep.gap < 1e-7                                  # LP optimality certificate
    v = check(prob, rep, mc_samples=0)
    assert v["ok"], v                                      # primal re-check passes
    assert v["max_violation"] > -1e-6                      # and is tight: dual bound is exact


def test_unconstrained_variability_matches_closed_form():
    G = np.array([[1.5, 0, 0, 0], [0, 1.2, 0, 0.6], [0, 0, 1.0, 0.6]])
    arch = Architecture([Model("S", np.zeros(3), G, "T", gen_teams=["DB", "Auth", "GW", "Infra"])])
    w = np.array([2.0, 1.0, 1.0])
    prob = RepairProblem(arch, [Variability("v", w, eps=5.0, owner="P")], shrink_max=0.8)
    rep = prob.solve(CostModel("smallest"))
    spread = np.abs(w @ (arch.G * rep.sigma)).sum()
    assert spread == pytest.approx(5.0, abs=1e-7)
    assert check(prob, rep)["ok"]


def _webshop_like():
    G = np.array([[1.5, 0, 0, 0], [0, 1.2, 0, 0.6], [0, 0, 1.0, 0.6]])
    arch = Architecture([Model("S", np.zeros(3), G, "T", gen_teams=["Database", "Auth", "Gateway", "Infra"])])
    reqs = [Variability("checkout", np.array([2.0, 1, 1]), 5.0, "Product"),
            Variability("login", np.array([0.0, 1, 1]), 3.0, "Product")]
    return RepairProblem(arch, reqs, shrink_max=np.array([0.8, 0.8, 0.8, 0.5]))


def test_reproduces_v0_prototype():
    prob = _webshop_like()
    prices = {"Database": {"shrink": 8000}, "Auth": {"shrink": 500},
              "Gateway": {"shrink": 2000}, "Infra": {"shrink": 6000}}
    rep = prob.solve(CostModel("unit", prices))
    assert rep.objective == pytest.approx(1360.0, abs=1e-6)
    assert rep.main_team() == "Auth"


def test_price_of_restriction_nonnegative_and_frozen_team_idle():
    prob = _webshop_like()
    cost = CostModel("unit", {"Auth": {"shrink": 500}, "Database": {"shrink": 8000}})
    for team, por in por_table(prob, cost).items():
        assert por >= -1e-7
        rep = prob.solve(cost, frozen={team})
        if rep.feasible:
            assert rep.team_abs.get(team, 0.0) == pytest.approx(0.0, abs=1e-9)


def test_pwl_single_piece_equals_unit():
    prob = _webshop_like()
    prices = {"Auth": {"shrink": 3.0}}
    unit = prob.solve(CostModel("unit", prices))
    pwl = prob.solve(CostModel("pwl", prices, pwl=[(0.0, 1.0)]))
    assert pwl.objective == pytest.approx(unit.objective, abs=1e-7)


def test_pwl_convexity_spreads_load():
    prob = _webshop_like()
    steep = CostModel("pwl", {}, pwl=[(0.0, 1.0), (0.3, 50.0)])   # >0.3 ms per team gets very expensive
    rep = prob.solve(steep)
    assert len(rep.changed_teams()) >= 2


def test_all_frozen_is_infeasible():
    prob = _webshop_like()
    rep = prob.solve(CostModel("smallest"), frozen={"Database", "Auth", "Gateway", "Infra"})
    assert not rep.feasible


def iris_s1(delta=0.0):
    src = Model("Source", np.array([100.0, 50.0]) + delta * 2.0, np.diag([2.0, 2.0]), "SensorA",
                gen_teams=["SensorA", "SensorB"])
    tgt = BoxTarget(np.array([100.3, 50.3]), np.array([2.1, 2.1]), "Target",
                    shift_max=5, widen_max=1.0)
    return Architecture([src]), tgt


def test_overlap_and_containment_on_iris_s1():
    arch, tgt = iris_s1()
    ov = RepairProblem(arch, [Overlap("ov", np.eye(2), tgt)])
    assert requirement_holds(ov)                           # nominal S1 overlaps
    cont = RepairProblem(arch, [BoxContainment("ct", np.eye(2), tgt)], shift_max=5)
    assert not requirement_holds(cont)                     # but is not contained
    rep = cont.solve(CostModel("smallest"))
    assert rep.feasible and check(cont, rep)["ok"]
    arch2, tgt2 = iris_s1(delta=3.0)                       # shifted 3 generator widths away
    ov2 = RepairProblem(arch2, [Overlap("ov", np.eye(2), tgt2)], shift_max=10)
    assert not requirement_holds(ov2)
    rep2 = ov2.solve(CostModel("smallest"))
    assert rep2.feasible and check(ov2, rep2)["ok"]


def test_mapping_propagation_matches_manual():
    a = Model("A", [1.0], [[1.0]], "TA")
    b = Model("B", [0.5], [[0.2]], "TB")
    arch = Architecture([b, a], [Mapping("A", "B", np.array([[2.0]]), np.array([0.1]))])
    xb = arch.var("B.x0")
    assert arch.c[xb] == pytest.approx(2 * 1.0 + 0.1 + 0.5)
    assert np.abs(arch.G[xb]).tolist() == pytest.approx([2.0, 0.2])
