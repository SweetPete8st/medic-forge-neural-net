"""Deterministic compiler: explicit scalar computation graph -> layered ReLU network with error bounds.

Why this exists. Hand-writing weights (exp0) works for a 4-variable model and does not scale to a
15-variable physiology with dozens of products and nonlinear functions. A compiler makes "the patient
brain as a network" reproducible by construction: the same program gives the same weights, byte for
byte, and every hidden unit is named in the manifest so a reviewer can trace an error to a node.

Representation. Every node is a scalar. Values can be negative, so a node materialised in a hidden
layer is stored as a canonical pair (relu(v), relu(-v)) and read back as their difference. Affine
nodes cost nothing until they are carried across a layer. Nonlinear nodes:
  relu(x)                 1 unit, exact
  max/min/clip            1-2 units, exact
  gate(x, g)  x*g         1 unit, exact when 0 <= x <= M and g in {0,1}
  step(x > t)             2 units (steep clip), exact outside a band of width 1/sharpness around t
  fn(x, f, knots)         K-1 units, piecewise-linear interpolant, exact at knots
  mul(a, b)               2 squaring interpolants, ab = (sq(a+b) - sq(a-b))/4, error <= h^2/8 * scale
Interval arithmetic propagates ranges so knots and gates are sized automatically.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .layers import ReLUNet


@dataclass
class Node:
    id: int
    kind: str
    name: str
    deps: List["Node"]
    lo: float
    hi: float
    params: dict = field(default_factory=dict)

    def __repr__(self):
        return f"{self.name}#{self.id}[{self.lo:.3g},{self.hi:.3g}]"


class Graph:
    def __init__(self):
        self.nodes: List[Node] = []
        self.inputs: List[Node] = []
        self.outputs: List[Tuple[str, Node]] = []

    # ------------------------------------------------------------ construction
    def _add(self, kind, name, deps, lo, hi, **params) -> Node:
        n = Node(len(self.nodes), kind, name, list(deps), float(lo), float(hi), params)
        self.nodes.append(n)
        return n

    def input(self, name: str, lo: float, hi: float) -> Node:
        n = self._add("input", name, [], lo, hi, index=len(self.inputs))
        self.inputs.append(n)
        return n

    def const(self, c: float, name: str = "const") -> Node:
        return self._add("affine", name, [], c, c, terms=[], bias=float(c))

    def affine(self, terms: Sequence[Tuple[Node, float]], bias: float = 0.0, name: str = "affine") -> Node:
        lo = hi = bias
        for n, a in terms:
            lo += min(a * n.lo, a * n.hi); hi += max(a * n.lo, a * n.hi)
        return self._add("affine", name, [n for n, _ in terms], lo, hi, terms=[(n, float(a)) for n, a in terms], bias=float(bias))

    def add(self, a: Node, b: Node, name="add") -> Node: return self.affine([(a, 1.0), (b, 1.0)], 0.0, name)
    def sub(self, a: Node, b: Node, name="sub") -> Node: return self.affine([(a, 1.0), (b, -1.0)], 0.0, name)
    def scale(self, a: Node, k: float, bias: float = 0.0, name="scale") -> Node: return self.affine([(a, k)], bias, name)

    def relu(self, x: Node, name="relu") -> Node:
        return self._add("relu", name, [x], max(x.lo, 0.0), max(x.hi, 0.0))

    def max(self, a: Node, b: Node, name="max") -> Node:
        return self._add("max", name, [a, b], max(a.lo, b.lo), max(a.hi, b.hi))

    def min(self, a: Node, b: Node, name="min") -> Node:
        return self._add("min", name, [a, b], min(a.lo, b.lo), min(a.hi, b.hi))

    def clip(self, x: Node, lo: float, hi: float, name="clip") -> Node:
        return self._add("clip", name, [x], max(lo, min(x.lo, hi)), max(lo, min(x.hi, hi)), clo=float(lo), chi=float(hi))

    def gate(self, x: Node, g: Node, name="gate") -> Node:
        """x * g for g in {0,1}. One unit relu(x - M(1-g)) when x >= 0 (exact); two units
        relu(x + Mg - M) - relu(-x + Mg - M) when the interval allows negative x (exact for |x| <= M)."""
        M = max(abs(x.lo), abs(x.hi), 1e-9)
        signed = x.lo < -1e-12
        return self._add("gate", name, [x, g], min(x.lo, 0.0), max(x.hi, 0.0), M=M, signed=signed)

    def step(self, x: Node, thresh: float, sharpness: float = 1e4, name="step") -> Node:
        """indicator(x > thresh) as a steep clip: exact when |x - thresh| >= 1/sharpness."""
        return self._add("step", name, [x], 0.0, 1.0, thresh=float(thresh), sharp=float(sharpness))

    def fn(self, x: Node, f: Callable[[float], float], knots: Optional[Sequence[float]] = None,
           n_knots: int = 48, spacing: str = "uniform", name="fn") -> Node:
        if knots is None:
            lo, hi = x.lo, x.hi
            if hi <= lo: hi = lo + 1e-6
            if spacing == "log":
                assert lo > 0, "log spacing needs lo > 0"
                knots = list(np.exp(np.linspace(math.log(lo), math.log(hi), n_knots)))
            else:
                knots = list(np.linspace(lo, hi, n_knots))
        knots = sorted(set(float(k) for k in knots))
        vals = [f(k) for k in knots]
        return self._add("fn", name, [x], min(vals), max(vals), f=f, knots=knots)

    def mul(self, a: Node, b: Node, eps_rel: float = 1e-3, name="mul") -> Node:
        """a*b via squaring interpolants with inputs scaled to about [-1,1].
        Absolute error bound ~ eps_rel * scale_a * scale_b."""
        sa = max(abs(a.lo), abs(a.hi), 1e-9); sb = max(abs(b.lo), abs(b.hi), 1e-9)
        h = math.sqrt(8.0 * eps_rel)
        n = int(math.ceil(4.0 / h)) + 1          # knots over [-2, 2]
        knots = list(np.linspace(-2.0, 2.0, n))
        cands = [a.lo * b.lo, a.lo * b.hi, a.hi * b.lo, a.hi * b.hi]
        return self._add("mul", name, [a, b], min(cands), max(cands), sa=sa, sb=sb, knots=knots)

    def log(self, x: Node, floor: float, n_knots: int = 96, name="log") -> Node:
        """ln(max(x, floor)) with log-spaced knots (relative error control). x must be bounded above."""
        xf = self.max(x, self.const(floor), name=f"{name}-floor")
        xf.lo = max(xf.lo, floor)
        return self.fn(xf, math.log, n_knots=n_knots, spacing="log", name=name)

    def exp(self, x: Node, n_knots: int = 256, name="exp") -> Node:
        return self.fn(x, math.exp, n_knots=n_knots, spacing="uniform", name=name)

    def mul_pos(self, factors: Sequence[Tuple[Node, float]], scale: float = 1.0, n_knots_exp: int = 256, name="mulpos") -> Node:
        """Product of positive bounded factors via logs: scale * prod(max(x_i, floor_i)).
        Relative error ~ h^2/8 of the exp interpolant plus log chord errors; good for wide ranges."""
        terms = [(self.log(x, floor, name=f"{name}-log{i}"), 1.0) for i, (x, floor) in enumerate(factors)]
        s = self.affine(terms, math.log(scale) if scale > 0 else 0.0, name=f"{name}-logsum")
        return self.exp(s, n_knots=n_knots_exp, name=name)

    def output(self, node: Node, name: str) -> None:
        self.outputs.append((name, node))

    # ------------------------------------------------------------ reference evaluation (float, exact graph semantics)
    def evaluate(self, x: Sequence[float]) -> Dict[str, float]:
        vals: Dict[int, float] = {}
        for n in self.nodes:
            k = n.kind
            if k == "input": v = float(x[n.params["index"]])
            elif k == "affine": v = n.params["bias"] + sum(a * vals[d.id] for d, a in n.params["terms"])
            elif k == "relu": v = max(vals[n.deps[0].id], 0.0)
            elif k == "max": v = max(vals[n.deps[0].id], vals[n.deps[1].id])
            elif k == "min": v = min(vals[n.deps[0].id], vals[n.deps[1].id])
            elif k == "clip": v = min(max(vals[n.deps[0].id], n.params["clo"]), n.params["chi"])
            elif k == "gate": v = vals[n.deps[0].id] * vals[n.deps[1].id]
            elif k == "step": v = 1.0 if vals[n.deps[0].id] > n.params["thresh"] else 0.0
            elif k == "fn": v = n.params["f"](vals[n.deps[0].id])
            elif k == "mul": v = vals[n.deps[0].id] * vals[n.deps[1].id]
            else: raise ValueError(k)
            vals[n.id] = v
        return {name: vals[node.id] for name, node in self.outputs}


# ---------------------------------------------------------------- compilation
class _Layering:
    """Builds layers of units. A 'readout' is an affine expression over units of one layer."""
    def __init__(self, n_inputs: int):
        self.layers: List[List[Tuple[Dict[int, float], float, str]]] = [[]]   # layer 0 = inputs (no units)
        self.n_inputs = n_inputs
        self.readouts: Dict[Tuple[int, int], Tuple[Dict[int, float], float]] = {}
        self.ready: Dict[int, int] = {}      # node id -> first layer at which a readout exists
        self.manifest: List[str] = []

    def width(self, L: int) -> int:
        return self.n_inputs if L == 0 else len(self.layers[L])

    def _ensure_layer(self, L: int):
        while len(self.layers) <= L:
            self.layers.append([])

    def unit(self, L: int, coeffs: Dict[int, float], bias: float, label: str) -> int:
        """Add a ReLU unit at layer L whose pre-activation is coeffs over layer L-1 units + bias."""
        self._ensure_layer(L)
        self.layers[L].append((dict(coeffs), float(bias), label))
        idx = len(self.layers[L]) - 1
        self.manifest.append(f"L{L}[{idx}] {label}")
        return idx

    def set_readout(self, node: Node, L: int, coeffs: Dict[int, float], bias: float):
        self.readouts[(node.id, L)] = (dict(coeffs), float(bias))
        if node.id not in self.ready or L < self.ready[node.id]:
            self.ready[node.id] = L

    def readout_at(self, node: Node, L: int) -> Tuple[Dict[int, float], float]:
        """Affine expression for node over layer-L units, carrying through pairs if needed."""
        if (node.id, L) in self.readouts:
            return self.readouts[(node.id, L)]
        r0 = self.ready[node.id]
        assert L > r0, f"node {node} not available at layer {L} (ready at {r0})"
        c, b = self.readout_at(node, L - 1)
        p = self.unit(L, c, b, f"carry+ {node.name}#{node.id}")
        n = self.unit(L, {k: -v for k, v in c.items()}, -b, f"carry- {node.name}#{node.id}")
        self.set_readout(node, L, {p: 1.0, n: -1.0}, 0.0)
        return self.readouts[(node.id, L)]


def _combine(parts: Sequence[Tuple[Dict[int, float], float, float]]) -> Tuple[Dict[int, float], float]:
    """Sum of (coeffs, bias) * weight."""
    out: Dict[int, float] = {}; bias = 0.0
    for c, b, w in parts:
        for k, v in c.items():
            out[k] = out.get(k, 0.0) + w * v
        bias += w * b
    return out, bias


def compile_graph(g: Graph) -> Tuple[ReLUNet, dict]:
    lay = _Layering(len(g.inputs))
    for n in g.inputs:
        lay.set_readout(n, 0, {n.params["index"]: 1.0}, 0.0)

    def place(n: Node):
        if n.id in lay.ready:
            return
        for d in n.deps:
            place(d)
        k = n.kind
        if k == "affine":
            L = max([lay.ready[d.id] for d in n.deps], default=0)
            parts = []
            for d, a in n.params["terms"]:
                c, b = lay.readout_at(d, L); parts.append((c, b, a))
            c, b = _combine(parts); b += n.params["bias"]
            lay.set_readout(n, L, c, b); return
        if k == "relu":
            x = n.deps[0]; L = lay.ready[x.id]; c, b = lay.readout_at(x, L)
            u = lay.unit(L + 1, c, b, f"relu {n.name}#{n.id}")
            lay.set_readout(n, L + 1, {u: 1.0}, 0.0); return
        if k in ("max", "min"):
            a, bnode = n.deps; L = max(lay.ready[a.id], lay.ready[bnode.id])
            ca, ba = lay.readout_at(a, L); cb, bb = lay.readout_at(bnode, L)
            # max(a,b) = a + relu(b - a); min(a,b) = a - relu(a - b)
            if k == "max":
                cd, bd = _combine([(cb, bb, 1.0), (ca, ba, -1.0)]); sign = 1.0
            else:
                cd, bd = _combine([(ca, ba, 1.0), (cb, bb, -1.0)]); sign = -1.0
            u = lay.unit(L + 1, cd, bd, f"{k} {n.name}#{n.id}")
            pa, qa = lay.unit(L + 1, ca, ba, f"{k}+ {n.name}#{n.id}"), lay.unit(L + 1, {kk: -v for kk, v in ca.items()}, -ba, f"{k}- {n.name}#{n.id}")
            lay.set_readout(n, L + 1, {pa: 1.0, qa: -1.0, u: sign}, 0.0); return
        if k == "clip":
            x = n.deps[0]; L = lay.ready[x.id]; c, b = lay.readout_at(x, L)
            lo, hi = n.params["clo"], n.params["chi"]
            u1 = lay.unit(L + 1, c, b - lo, f"clip-lo {n.name}#{n.id}")
            u2 = lay.unit(L + 1, c, b - hi, f"clip-hi {n.name}#{n.id}")
            lay.set_readout(n, L + 1, {u1: 1.0, u2: -1.0}, lo); return
        if k == "gate":
            x, gg = n.deps; L = max(lay.ready[x.id], lay.ready[gg.id]); M = n.params["M"]
            cx, bx = lay.readout_at(x, L); cg, bg = lay.readout_at(gg, L)
            c, b = _combine([(cx, bx, 1.0), (cg, bg, M)]); b -= M
            u = lay.unit(L + 1, c, b, f"gate {n.name}#{n.id}")
            if n.params.get("signed"):
                c2, b2 = _combine([(cx, bx, -1.0), (cg, bg, M)]); b2 -= M
                u2 = lay.unit(L + 1, c2, b2, f"gate- {n.name}#{n.id}")
                lay.set_readout(n, L + 1, {u: 1.0, u2: -1.0}, 0.0); return
            lay.set_readout(n, L + 1, {u: 1.0}, 0.0); return
        if k == "step":
            x = n.deps[0]; L = lay.ready[x.id]; c, b = lay.readout_at(x, L); s = n.params["sharp"]; t = n.params["thresh"]
            cs = {kk: s * v for kk, v in c.items()}
            u1 = lay.unit(L + 1, cs, s * (b - t), f"step0 {n.name}#{n.id}")
            u2 = lay.unit(L + 1, cs, s * (b - t) - 1.0, f"step1 {n.name}#{n.id}")
            lay.set_readout(n, L + 1, {u1: 1.0, u2: -1.0}, 0.0); return
        if k == "fn":
            x = n.deps[0]; L = lay.ready[x.id]; c, b = lay.readout_at(x, L)
            ks, f = n.params["knots"], n.params["f"]
            slopes = [(f(ks[i + 1]) - f(ks[i])) / (ks[i + 1] - ks[i]) for i in range(len(ks) - 1)]
            units: Dict[int, float] = {}; prev = 0.0
            for j, s in enumerate(slopes):
                u = lay.unit(L + 1, c, b - ks[j], f"fn-knot{j} {n.name}#{n.id}")
                units[u] = s - prev; prev = s
            # left of k0 the interpolant is constant f(k0); extrapolate linearly there too for safety
            lay.set_readout(n, L + 1, units, float(f(ks[0]))); return
        if k == "mul":
            a, bnode = n.deps; L = max(lay.ready[a.id], lay.ready[bnode.id]); sa, sb, ks = n.params["sa"], n.params["sb"], n.params["knots"]
            ca, ba = lay.readout_at(a, L); cb, bb = lay.readout_at(bnode, L)
            # scaled s = a/sa + b/sb, d = a/sa - b/sb, both in [-2, 2]
            cs, bs = _combine([(ca, ba, 1.0 / sa), (cb, bb, 1.0 / sb)])
            cd, bd = _combine([(ca, ba, 1.0 / sa), (cb, bb, -1.0 / sb)])
            sq = lambda z: z * z
            slopes = [(sq(ks[i + 1]) - sq(ks[i])) / (ks[i + 1] - ks[i]) for i in range(len(ks) - 1)]
            ro: Dict[int, float] = {}
            # ab = sa*sb*(sq(s) - sq(d))/4 ; sq(k0) = 4 for both so constants cancel
            for sign, (cc, bb_) in ((1.0, (cs, bs)), (-1.0, (cd, bd))):
                prev = 0.0
                for j, s in enumerate(slopes):
                    u = lay.unit(L + 1, cc, bb_ - ks[j], f"mul-{'s' if sign > 0 else 'd'}{j} {n.name}#{n.id}")
                    ro[u] = sign * (s - prev) * sa * sb / 4.0; prev = s
            lay.set_readout(n, L + 1, ro, 0.0); return
        raise ValueError(k)

    for _, node in g.outputs:
        place(node)
    L_out = max(lay.ready[node.id] for _, node in g.outputs)
    # final linear layer: outputs as readouts at L_out (carry if needed)
    out_rows = [lay.readout_at(node, L_out) for _, node in g.outputs]
    # ---- assemble weight matrices
    net = ReLUNet()
    for L in range(1, len(lay.layers)):
        W = np.zeros((len(lay.layers[L]), lay.width(L - 1))); b = np.zeros(len(lay.layers[L]))
        for i, (coeffs, bias, _) in enumerate(lay.layers[L]):
            for k, v in coeffs.items(): W[i, k] = v
            b[i] = bias
        if W.shape[0] == 0:   # empty layer (should not happen) -> skip
            continue
        net.add(W, b)
    Wo = np.zeros((len(out_rows), lay.width(L_out))); bo = np.zeros(len(out_rows))
    for i, (coeffs, bias) in enumerate(out_rows):
        for k, v in coeffs.items(): Wo[i, k] = v
        bo[i] = bias
    net.add(Wo, bo)
    widths = [lay.width(L) for L in range(len(lay.layers))]
    nnz = int(sum((W != 0).sum() + (bb != 0).sum() for W, bb in net.layers))
    manifest = {"widths": widths + [len(out_rows)], "n_layers": len(net.layers), "dense_params": net.n_params,
                "nonzero_params": nnz, "units": lay.manifest, "outputs": [name for name, _ in g.outputs],
                "inputs": [n.name for n in g.inputs]}
    return net, manifest
