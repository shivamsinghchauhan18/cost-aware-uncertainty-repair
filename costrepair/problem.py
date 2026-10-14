"""Cost-aware, certified repair of uncertainty requirements.

Decision variables ("levers"), each owned by one team:
  * generator scale  sigma_k = 1 + up_k - down_k   (shrink or widen an uncertainty source)
  * center shift     dc_i = pos_i - neg_i           (recalibrate a model variable)
  * target shift / target scale on box targets      (revise the other model / the requirement)
  * relaxation       r_j >= 0                       (loosen a requirement bound)

Requirements on the global set X = { c + T dc + G diag(sigma) xi : A xi = b, |xi| <= 1 }:
  * Bound(w, beta):            max_{x in X} w^T x <= beta + r                (certified containment)
  * Variability(w, eps):       max w^T x - min w^T x <= 2 (eps + r)          ("may vary by at most")
  * BoxContainment(M, target): M x in target box for every x in X           (all realizations consistent)
  * Overlap(M, target):        exists x in X with M x in target box          (joint consistency, Def. 2
                                                                             of Mäkelburg et al. 2026)

Support functions of the constrained part are written through LP duality:
  max_xi sum_k a_k sigma_k xi_k  s.t. A xi = b, |xi| <= 1
    = min_{lam, mu+, mu- >= 0}  b^T lam + 1^T (mu+ + mu-)  s.t.  A^T lam + mu+ - mu- = a o sigma
The right-hand side is linear in sigma, so "requirement holds after repair" is a set of linear
constraints over (levers, lam, mu). Any feasible (lam, mu) is a proof (weak duality), so every
repair returned is certified, and the overall optimum is certified by a zero LP duality gap.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .architecture import Architecture
from .lp import INF, LP

KINDS = ("shrink", "widen", "shift", "relax", "target_shift", "target_widen", "target_shrink")


# --------------------------------------------------------------------------- requirements
@dataclass
class Bound:
    name: str
    w: np.ndarray            # over global x
    beta: float
    owner: str               # team that owns the requirement (may relax it)
    relax_max: float = 0.0   # 0 = requirement is fixed
    scale: float = 1.0       # unit used to normalise a relaxation in the "smallest" baseline


@dataclass
class Variability:
    name: str
    w: np.ndarray
    eps: float
    owner: str
    relax_max: float = 0.0
    scale: float = 1.0


@dataclass
class BoxTarget:
    """Target box {y : |y - ct| <= ht * tau}, revisable by its owner (shift ct, scale tau)."""
    ct: np.ndarray
    ht: np.ndarray
    owner: str
    shift_max: float = 0.0          # absolute
    widen_max: float = 0.0          # relative (tau <= 1 + widen_max)
    shrink_max: float = 0.0         # relative (tau >= 1 - shrink_max)


@dataclass
class BoxContainment:
    name: str
    M: np.ndarray            # (d, n) output map over global x
    target: BoxTarget


@dataclass
class Overlap:
    name: str
    M: np.ndarray
    target: BoxTarget


# --------------------------------------------------------------------------- cost models
@dataclass
class CostModel:
    """How a repair is priced.

    kind = "smallest"      : sum of relative changes, every team equal  (current method, l1)
           "smallest_linf" : largest relative change                      (current method, l_inf)
           "unit"          : price[team][lever kind] per absolute unit changed
           "proportional"  : price[team][lever kind] per relative unit changed
           "pwl"           : convex piecewise-linear function of each team's unit-priced effort
    prices: team -> {lever kind: price}; a missing kind uses `default_price`;
            price = inf means "this team cannot do that".
    pwl: list of (breakpoint, slope) with increasing slopes, applied per team.
    """
    kind: str = "smallest"
    prices: dict[str, dict[str, float]] = field(default_factory=dict)
    default_price: float = 1.0
    pwl: list[tuple[float, float]] | None = None

    def price(self, team: str, lever: str) -> float:
        return self.prices.get(team, {}).get(lever, self.default_price)


@dataclass
class Lever:
    var: int
    team: str
    kind: str
    abs_measure: float       # change in absolute units per unit of the variable
    rel_measure: float       # change in relative units per unit of the variable
    label: str


@dataclass
class Repair:
    status: str
    objective: float = INF
    gap: float = np.nan
    sigma: np.ndarray | None = None
    dc: np.ndarray | None = None
    relax: dict[str, float] = field(default_factory=dict)
    targets: dict[str, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    team_abs: dict[str, float] = field(default_factory=dict)
    team_rel: dict[str, float] = field(default_factory=dict)
    team_unit_cost: dict[str, float] = field(default_factory=dict)
    shadow: dict[str, float] = field(default_factory=dict)
    lever_values: list[tuple[str, str, str, float]] = field(default_factory=list)

    @property
    def feasible(self) -> bool:
        return self.status == "optimal"

    def changed_teams(self, tol: float = 1e-7) -> set[str]:
        return {t for t, v in self.team_rel.items() if v > tol}

    def main_team(self) -> str | None:
        if not self.feasible or not self.team_rel or max(self.team_rel.values()) <= 1e-9:
            return None
        return max(self.team_rel, key=self.team_rel.get)

    @property
    def unit_cost(self) -> float:
        return sum(self.team_unit_cost.values()) if self.feasible else INF


# --------------------------------------------------------------------------- problem
@dataclass
class RepairProblem:
    arch: Architecture
    requirements: list
    shrink_max: float | np.ndarray = 0.9      # sigma >= 1 - shrink_max
    widen_max: float | np.ndarray = 0.0       # sigma <= 1 + widen_max
    shift_max: float | np.ndarray = 0.0       # |dc_i| <= shift_max (absolute)
    adjustable_gens: set[int] | None = None   # None = all
    shiftable_vars: set[int] | None = None    # None = all with shift_max > 0

    # ----------------------------------------------------------------- build
    def _vec(self, v, size):
        return np.full(size, float(v)) if np.isscalar(v) else np.asarray(v, float)

    def build(self, cost: CostModel, frozen: set[str] = frozenset()):
        a = self.arch
        lp = LP()
        levers: list[Lever] = []
        shr = self._vec(self.shrink_max, a.m)
        wid = self._vec(self.widen_max, a.m)
        sft = self._vec(self.shift_max, a.n)
        size = a.local_gen_size()
        vscale = a.local_var_scale()
        has_overlap = any(isinstance(r, Overlap) for r in self.requirements)

        def lever(team, kind, ub, absm, relm, label):
            p = cost.price(team, kind) if cost.kind in ("unit", "proportional", "pwl") else 1.0
            if team in frozen or not np.isfinite(p) or ub <= 0:
                ub = 0.0
            i = lp.var(label, 0.0, ub)
            levers.append(Lever(i, team, kind, absm, relm, label))
            return i

        # generator scales
        up, dn = {}, {}
        for k in range(a.m):
            adj = self.adjustable_gens is None or k in self.adjustable_gens
            if has_overlap and k in a.constrained_gens:
                adj = False     # keep A xi = b linear in the overlap witness
            t = a.gen_team[k]
            up[k] = lever(t, "widen", wid[k] if adj else 0, size[k], 1.0, f"widen {a.gen_names[k]}")
            dn[k] = lever(t, "shrink", shr[k] if adj else 0, size[k], 1.0, f"shrink {a.gen_names[k]}")

        # center shifts (local variables; propagated by T)
        pos, neg = {}, {}
        for i in range(a.n):
            ok = (self.shiftable_vars is None or i in self.shiftable_vars) and sft[i] > 0
            t = a.var_team[i]
            pos[i] = lever(t, "shift", sft[i] if ok else 0, 1.0, 1 / vscale[i], f"shift+ {a.var_names[i]}")
            neg[i] = lever(t, "shift", sft[i] if ok else 0, 1.0, 1 / vscale[i], f"shift- {a.var_names[i]}")

        def center_expr(wrow):
            """w^T (c + T dc) as (expr, const)."""
            wt = wrow @ a.T
            e = {}
            for i in range(a.n):
                if wt[i] != 0:
                    e[pos[i]] = e.get(pos[i], 0) + wt[i]
                    e[neg[i]] = e.get(neg[i], 0) - wt[i]
            return e, float(wrow @ a.c)

        C = sorted(a.constrained_gens)

        def support_expr(wrow):
            """Upper bound on max_xi (w^T G diag(sigma) xi), exact at optimum: (expr, const)."""
            av = a.G.T @ wrow
            e, const = {}, 0.0
            for k in range(a.m):
                if k in a.constrained_gens:
                    continue
                if av[k] != 0:          # |a_k| sigma_k, sigma_k >= 0
                    s = abs(av[k])
                    const += s
                    e[up[k]] = e.get(up[k], 0) + s
                    e[dn[k]] = e.get(dn[k], 0) - s
            if C and np.any(av[C] != 0):
                lam = [lp.var("lam", -INF, INF) for _ in range(a.A.shape[0])]
                for r, li in enumerate(lam):
                    e[li] = e.get(li, 0) + a.b[r]
                for k in C:
                    mp, mn = lp.var("mu+"), lp.var("mu-")
                    e[mp] = e.get(mp, 0) + 1.0
                    e[mn] = e.get(mn, 0) + 1.0
                    row = {li: a.A[r, k] for r, li in enumerate(lam) if a.A[r, k] != 0}
                    row[mp] = row.get(mp, 0) + 1.0
                    row[mn] = row.get(mn, 0) - 1.0
                    row[up[k]] = row.get(up[k], 0) - av[k]
                    row[dn[k]] = row.get(dn[k], 0) + av[k]
                    lp.eq(row, av[k])           # A^T lam + mu+ - mu- = a_k (1 + up - down)
            return e, const

        def add(e1, e2, s=1.0):
            out = dict(e1)
            for k, v in e2.items():
                out[k] = out.get(k, 0) + s * v
            return out

        tgt_vars = {}

        def target_levers(req, d):
            T_ = req.target
            ts = [(lever(T_.owner, "target_shift", T_.shift_max, 1.0, 1 / max(T_.ht[i], 1e-12), f"tshift+ {req.name}[{i}]"),
                   lever(T_.owner, "target_shift", T_.shift_max, 1.0, 1 / max(T_.ht[i], 1e-12), f"tshift- {req.name}[{i}]"))
                  for i in range(d)]
            tw = [(lever(T_.owner, "target_widen", T_.widen_max, T_.ht[i], 1.0, f"twiden {req.name}[{i}]"),
                   lever(T_.owner, "target_shrink", T_.shrink_max, T_.ht[i], 1.0, f"tshrink {req.name}[{i}]"))
                  for i in range(d)]
            tgt_vars[req.name] = (ts, tw)
            return ts, tw

        relax_vars = {}
        for req in self.requirements:
            if isinstance(req, Bound):
                r = lever(req.owner, "relax", req.relax_max, 1.0, 1 / req.scale, f"relax {req.name}")
                relax_vars[req.name] = r
                ce, cc = center_expr(req.w)
                se, sc = support_expr(req.w)
                e = add(add(ce, se), {r: -1.0})
                lp.le(e, req.beta - cc - sc, tag=req.name)
            elif isinstance(req, Variability):
                r = lever(req.owner, "relax", req.relax_max, 1.0, 1 / req.scale, f"relax {req.name}")
                relax_vars[req.name] = r
                s1, c1 = support_expr(req.w)
                s2, c2 = support_expr(-req.w)
                e = add(add(s1, s2), {r: -2.0})
                lp.le(e, 2 * req.eps - c1 - c2, tag=req.name)
            elif isinstance(req, BoxContainment):
                d = req.M.shape[0]
                ts, tw = target_levers(req, d)
                for i in range(d):
                    for sgn in (+1.0, -1.0):
                        wrow = sgn * req.M[i]
                        ce, cc = center_expr(wrow)
                        se, sc = support_expr(wrow)
                        # sgn*(y_i) <= sgn*(ct_i + dct_i) + ht_i * (1 + w_up - w_dn)
                        e = add(ce, se)
                        e = add(e, {ts[i][0]: -sgn, ts[i][1]: sgn})
                        e = add(e, {tw[i][0]: -req.target.ht[i], tw[i][1]: req.target.ht[i]})
                        lp.le(e, sgn * req.target.ct[i] + req.target.ht[i] - cc - sc, tag=req.name)
            elif isinstance(req, Overlap):
                d = req.M.shape[0]
                ts, tw = target_levers(req, d)
                # witness eta_k = sigma_k xi_k with |eta_k| <= sigma_k  (and A eta = b on fixed gens)
                eta = [lp.var(f"eta{k}", -INF, INF) for k in range(a.m)]
                for k in range(a.m):
                    lp.le({eta[k]: 1.0, up[k]: -1.0, dn[k]: 1.0}, 1.0, tag="_witness")
                    lp.le({eta[k]: -1.0, up[k]: -1.0, dn[k]: 1.0}, 1.0, tag="_witness")
                for r_ in range(a.A.shape[0]):
                    lp.eq({eta[k]: a.A[r_, k] for k in range(a.m) if a.A[r_, k] != 0}, a.b[r_])
                for i in range(d):
                    wrow = req.M[i]
                    ce, cc = center_expr(wrow)
                    gv = a.G.T @ wrow
                    ye = add(ce, {eta[k]: gv[k] for k in range(a.m) if gv[k] != 0})
                    for sgn in (+1.0, -1.0):
                        e = {k_: sgn * v for k_, v in ye.items()}
                        e = add(e, {ts[i][0]: -sgn, ts[i][1]: sgn})
                        e = add(e, {tw[i][0]: -req.target.ht[i], tw[i][1]: req.target.ht[i]})
                        lp.le(e, sgn * req.target.ct[i] + req.target.ht[i] - sgn * cc, tag=req.name)
            else:
                raise TypeError(type(req))

        # objective
        if cost.kind == "smallest":
            for lv in levers:
                lp.add_cost(lv.var, lv.rel_measure)
        elif cost.kind == "smallest_linf":
            t = lp.var("t", 0.0, INF, cost=1.0)
            for lv in levers:
                lp.le({lv.var: lv.rel_measure, t: -1.0}, 0.0, tag="_linf")
        elif cost.kind in ("unit", "proportional"):
            for lv in levers:
                m_ = lv.abs_measure if cost.kind == "unit" else lv.rel_measure
                p = cost.price(lv.team, lv.kind)
                lp.add_cost(lv.var, (p if np.isfinite(p) else 0.0) * m_)
        elif cost.kind == "pwl":
            pieces = cost.pwl or [(0.0, 1.0)]
            for team in sorted({lv.team for lv in levers}):
                z = lp.var(f"z[{team}]", 0.0, INF, cost=1.0)
                eff = {lv.var: (cost.price(team, lv.kind) if np.isfinite(cost.price(team, lv.kind)) else 0.0) * lv.abs_measure
                       for lv in levers if lv.team == team}
                # Continuous f with f(bp_0) = 0 and slope_l on [bp_l, bp_{l+1}];
                # convex (slopes increase) => f(E) = max_l (slope_l E + f(bp_l) - slope_l bp_l).
                affine, val = [], 0.0
                for idx, (bp, slope) in enumerate(pieces):
                    if idx > 0:
                        val += pieces[idx - 1][1] * (bp - pieces[idx - 1][0])
                    affine.append((slope, val - slope * bp))
                for slope, b0 in affine:
                    row = {k_: slope * v for k_, v in eff.items()}
                    row[z] = -1.0
                    lp.le(row, -b0, tag="_pwl")
        else:
            raise ValueError(cost.kind)
        return lp, levers, up, dn, pos, neg, relax_vars, tgt_vars

    # ----------------------------------------------------------------- solve
    def solve(self, cost: CostModel, frozen: set[str] = frozenset(),
              price_for_report: CostModel | None = None) -> Repair:
        lp, levers, up, dn, pos, neg, relax_vars, tgt_vars = self.build(cost, frozen)
        sol = lp.solve()
        if sol.status != "optimal":
            return Repair(status=sol.status)
        x = sol.x
        a = self.arch
        sigma = np.array([1 + x[up[k]] - x[dn[k]] for k in range(a.m)])
        dc = np.array([x[pos[i]] - x[neg[i]] for i in range(a.n)])
        rep = Repair(status="optimal", objective=sol.objective, gap=sol.gap, sigma=sigma, dc=dc)
        rep.relax = {n: float(x[v]) for n, v in relax_vars.items()}
        for name, (ts, tw) in tgt_vars.items():
            rep.targets[name] = (np.array([x[p] - x[q] for p, q in ts]),
                                 np.array([1 + x[p] - x[q] for p, q in tw]))
        pr = price_for_report or (cost if cost.kind in ("unit", "pwl") else None)
        for lv in levers:
            v = float(x[lv.var])
            rep.team_abs[lv.team] = rep.team_abs.get(lv.team, 0.0) + lv.abs_measure * v
            rep.team_rel[lv.team] = rep.team_rel.get(lv.team, 0.0) + lv.rel_measure * v
            p = pr.price(lv.team, lv.kind) if pr else 1.0
            rep.team_unit_cost[lv.team] = rep.team_unit_cost.get(lv.team, 0.0) + (p if np.isfinite(p) else 0.0) * lv.abs_measure * v
            if v > 1e-9:
                rep.lever_values.append((lv.team, lv.kind, lv.label, v))
        for tag, y in zip(lp.row_tags, sol.ineq_duals):
            if tag and not tag.startswith("_"):
                rep.shadow[tag] = rep.shadow.get(tag, 0.0) - float(y)
        return rep
