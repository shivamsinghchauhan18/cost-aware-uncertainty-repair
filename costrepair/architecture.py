"""Multi-model architectures with constrained-zonotope uncertainty.

Each model j owns state variables x_j and uncertainty sources (generators):
    x_j = sum_{p in parents(j)} F_pj x_p + f_pj  +  c_j + G_j xi_j,    A_j xi_j = b_j,  xi_j in [-1,1]
The parent terms are the *uncertainty mappings* (affine, acyclic). Stacking all
models in topological order gives one global constrained zonotope
    x = T (c + G_blk xi),  A_blk xi = b_blk
with T = (I - L)^{-1} the propagation matrix of the mapping graph L.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Model:
    name: str
    c: np.ndarray                         # (d,) local nominal values / additive offsets
    G: np.ndarray                         # (d, g) local generators
    team: str                             # team that owns (can revise) this model
    A: np.ndarray | None = None           # (q, g) constraint matrix on xi_j
    b: np.ndarray | None = None           # (q,)
    var_names: list[str] | None = None
    gen_names: list[str] | None = None
    gen_teams: list[str] | None = None    # override owner per generator (e.g. shared infra)

    def __post_init__(self):
        self.c = np.atleast_1d(np.asarray(self.c, float))
        self.G = np.asarray(self.G, float).reshape(len(self.c), -1)
        if self.var_names is None:
            self.var_names = [f"{self.name}.x{i}" for i in range(len(self.c))]
        if self.gen_names is None:
            self.gen_names = [f"{self.name}.g{k}" for k in range(self.G.shape[1])]
        if self.gen_teams is None:
            self.gen_teams = [self.team] * self.G.shape[1]


@dataclass
class Mapping:
    parent: str
    child: str
    F: np.ndarray                         # (d_child, d_parent)
    f: np.ndarray | None = None           # (d_child,)


@dataclass
class Architecture:
    models: list[Model]
    mappings: list[Mapping] = field(default_factory=list)

    def __post_init__(self):
        self._assemble()

    # -- global representation ---------------------------------------------------
    def _assemble(self):
        order = self._topo_order()
        self.models = [self.by_name[n] for n in order]
        offs, gofs, n, m = {}, {}, 0, 0
        for mdl in self.models:
            offs[mdl.name] = n
            gofs[mdl.name] = m
            n += len(mdl.c)
            m += mdl.G.shape[1]
        self.n, self.m = n, m
        self.var_offset, self.gen_offset = offs, gofs

        L = np.zeros((n, n))
        f_add = np.zeros(n)
        for mp in self.mappings:
            pc, po = self.by_name[mp.child], offs[mp.parent]
            co = offs[mp.child]
            F = np.asarray(mp.F, float).reshape(len(pc.c), -1)
            L[co:co + F.shape[0], po:po + F.shape[1]] += F
            if mp.f is not None:
                f_add[co:co + F.shape[0]] += np.asarray(mp.f, float)
        self.T = np.linalg.inv(np.eye(n) - L)       # propagation (L is strictly block-lower)

        self.c_local = np.concatenate([mdl.c for mdl in self.models]) + f_add
        G_blk = np.zeros((n, m))
        A_rows, b_rows = [], []
        for mdl in self.models:
            o, g = offs[mdl.name], gofs[mdl.name]
            G_blk[o:o + len(mdl.c), g:g + mdl.G.shape[1]] = mdl.G
            if mdl.A is not None and len(mdl.A):
                Ab = np.zeros((mdl.A.shape[0], m))
                Ab[:, g:g + mdl.G.shape[1]] = mdl.A
                A_rows.append(Ab)
                b_rows.append(np.asarray(mdl.b, float))
        self.G_local = G_blk
        self.G = self.T @ G_blk                       # global generators
        self.c = self.T @ self.c_local                # global center
        self.A = np.vstack(A_rows) if A_rows else np.zeros((0, m))
        self.b = np.concatenate(b_rows) if b_rows else np.zeros(0)
        self.var_names = [v for mdl in self.models for v in mdl.var_names]
        self.gen_names = [g for mdl in self.models for g in mdl.gen_names]
        self.gen_team = [t for mdl in self.models for t in mdl.gen_teams]
        self.var_team = [mdl.team for mdl in self.models for _ in mdl.c]
        self.constrained_gens = set(np.nonzero(np.abs(self.A).sum(axis=0) > 0)[0]) if len(self.A) else set()

    @property
    def by_name(self) -> dict[str, Model]:
        return {mdl.name: mdl for mdl in self.models}

    def _topo_order(self) -> list[str]:
        names = [mdl.name for mdl in self.models]
        parents = {nm: [mp.parent for mp in self.mappings if mp.child == nm] for nm in names}
        done, order = set(), []
        while len(order) < len(names):
            ready = [nm for nm in names if nm not in done and all(p in done for p in parents[nm])]
            if not ready:
                raise ValueError("uncertainty mappings must be acyclic")
            for nm in ready:
                done.add(nm)
                order.append(nm)
        return order

    def teams(self) -> list[str]:
        return sorted(set(self.gen_team) | set(self.var_team))

    def var(self, name: str) -> int:
        return self.var_names.index(name)

    def selector(self, *names: str) -> np.ndarray:
        """Row selector M so that M x picks the named global variables."""
        M = np.zeros((len(names), self.n))
        for r, nm in enumerate(names):
            M[r, self.var(nm)] = 1.0
        return M

    def local_gen_size(self) -> np.ndarray:
        """||g_k||_1 in the owning model's own units (absolute measure of a source)."""
        return np.abs(self.G_local).sum(axis=0)

    def local_var_scale(self) -> np.ndarray:
        """Half-width of each model variable's *own* uncertainty (relative unit for shifts)."""
        s = np.abs(self.G_local).sum(axis=1)
        return np.where(s > 0, s, 1.0)
