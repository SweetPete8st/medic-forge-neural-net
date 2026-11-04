import math, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from mfnn.wsp.compiler import Graph, compile_graph


def _check(g, samples, tol):
    net, man = compile_graph(g)
    worst = 0.0
    for x in samples:
        ref = g.evaluate(x); got = net.forward(np.array(x, dtype=float))
        for i, (name, _) in enumerate(g.outputs):
            worst = max(worst, abs(got[i] - ref[name]))
    assert worst <= tol, (worst, man["widths"])
    return net, man, worst


def test_exact_ops():
    g = Graph(); x = g.input("x", -3, 3); y = g.input("y", -2, 2)
    g.output(g.affine([(x, 2.0), (y, -1.0)], 0.5), "aff")
    g.output(g.relu(g.sub(x, y)), "relu")
    g.output(g.max(x, y), "max"); g.output(g.min(x, y), "min")
    g.output(g.clip(x, -1, 1), "clip")
    rng = np.random.default_rng(0)
    _check(g, rng.uniform([-3, -2], [3, 2], size=(500, 2)), 1e-12)


def test_gate_step_and_carry_depth():
    g = Graph(); x = g.input("x", 0, 10); flag = g.input("g", 0, 1)
    s = g.step(x, 5.0, sharpness=1e6)
    deep = g.relu(g.relu(g.relu(g.gate(x, flag))))       # forces carries of nothing but exercises depth
    g.output(g.gate(x, s), "x_if_gt5"); g.output(deep, "deep"); g.output(g.add(deep, x), "deep_plus_x")  # carry x across 3 layers
    rng = np.random.default_rng(1)
    xs = rng.uniform(0, 10, 300); gs = rng.integers(0, 2, 300)
    samples = [(a, b) for a, b in zip(xs, gs) if abs(a - 5.0) > 1e-3]
    _check(g, samples, 1e-7)   # steep clips lose ~sharpness*|x-t|*eps_machine


def test_fn_and_mul_bounds():
    g = Graph(); a = g.input("a", 0, 4); b = g.input("b", -1, 3)
    g.output(g.mul(a, b, eps_rel=1e-3), "ab")
    g.output(g.fn(a, lambda c: c / (1.0 + c), n_knots=64), "emax")
    g.output(g.fn(g.affine([(a, 1.0)], 0.1), lambda v: 1.0 / v, n_knots=96, spacing="log"), "recip")
    rng = np.random.default_rng(2)
    net, man, worst = _check(g, rng.uniform([0, -1], [4, 3], size=(400, 2)), 0.05)
    assert worst < 0.02, worst
    assert man["nonzero_params"] < man["dense_params"]


def test_recurrent_pk_cell_matches_code():
    """Compile the exp0 PK step through the compiler and compare with pk_code on a rollout."""
    sys.path.insert(0, str(ROOT / "experiments/exp0_pk"))
    import pk_code, yaml
    spec = yaml.safe_load((ROOT / "experiments/exp0_pk/pk_spec.yaml").read_text()); v = spec["variants"]["dev"]["typical"]
    r = pk_code.rates(v, 1.0)
    g = Graph()
    A_d = g.input("A_d", 0, 30); A_c = g.input("A_c", 0, 30); C_e = g.input("C_e", 0, 30); tau = g.input("tau", 0, 20000)
    u_d = g.input("u_d", 0, 30); u_c = g.input("u_c", 0, 30)
    Ad1 = g.affine([(A_d, r["a_d"]), (u_d, r["a_d"])], 0, "A_d'")
    Ac1 = g.affine([(A_d, r["a_c"] * (1 - r["a_d"])), (A_c, r["a_c"]), (u_d, r["a_c"] * (1 - r["a_d"])), (u_c, r["a_c"])], 0, "A_c'")
    Ce1 = g.affine([(C_e, 1 - r["alpha"]), (Ac1, r["alpha"])], 0, "C_e'")
    dose = g.step(g.add(u_d, u_c), 1e-9, sharpness=1e9)
    tau1 = g.relu(g.affine([(tau, 1.0), (dose, -1e7)], 1.0), "tau'")
    E = g.fn(Ce1, lambda c: v["Emax"] * c / (v["EC50"] + c), knots=[k * v["EC50"] for k in [0, .02, .05, .1, .15, .2, .3, .4, .5, .65, .8, 1, 1.25, 1.5, 2, 2.5, 3, 4, 5, 6.5, 8, 10, 13, 17, 22, 30]], name="E")
    for name, node in (("A_d'", Ad1), ("A_c'", Ac1), ("C_e'", Ce1), ("tau'", tau1), ("E", E)): g.output(node, name)
    net, man = compile_graph(g)
    doses = [(60, "depot", 2.5), (1260, "depot", 2.5), (2000, "central", 0.3)]
    ref = pk_code.rollout(v, doses, 3600)
    by_t = {}
    for t, route, amt in doses:
        d = by_t.setdefault(int(t), [0.0, 0.0]); d[0 if route == "depot" else 1] += float(amt)
    s = np.zeros(4); worst = 0.0
    for t in range(3600):
        u = by_t.get(t, [0.0, 0.0]); y = net.forward(np.array([s[0], s[1], s[2], s[3], u[0], u[1]])); s = y[:4]
        worst = max(worst, abs(y[4] - ref[t, 4]))
        assert abs(y[2] - ref[t, 2]) < 1e-9 and y[3] == ref[t, 3]
    assert worst < 0.01, worst
