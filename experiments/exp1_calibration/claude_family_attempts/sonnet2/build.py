import math, sys, numpy as np, yaml
sys.path.insert(0, '.')
from wsp_layers import ReLUNet, interpolant_rows
LN2 = math.log(2)
spec = yaml.safe_load(open('pk_spec_dev.yaml'))
CMAX = 12.0
def knots(M, ec, tol):
    ks = [0.0]
    while ks[-1] < CMAX:
        c = ks[-1]; f2 = 2*M*ec/(ec+c)**3
        ks.append(c + math.sqrt(8*tol/f2))
    return ks
def build(v):
    k = lambda h: LN2/(h*60)
    ad, ac, al = math.exp(-k(v['ka_half_min'])), math.exp(-k(v['ke_half_min'])), 1-math.exp(-k(v['keo_half_min']))
    ni = 6
    # layer 1 rows
    R = []; B = []
    def row(d, b=0.0):
        r = np.zeros(ni)
        for i, c in d.items(): r[i] = c
        R.append(r); B.append(b)
    row({0: ad, 4: ad})                                   # 0 Ad'
    # Ac' = ac*(Ac+uc+(1-ad)*(Ad+ud))
    row({1: ac, 5: ac, 0: ac*(1-ad), 4: ac*(1-ad)})        # 1 Ac'
    # Ce' = (1-al)Ce + al*Ac'
    row({2: 1-al, 1: al*ac, 5: al*ac, 0: al*ac*(1-ad), 4: al*ac*(1-ad)})  # 2 Ce'
    row({3: 1.0})                                          # 3 tau
    row({4: 1e3, 5: 1e3})                                  # 4 ReLU(1e3 s)
    row({4: 1e3, 5: 1e3}, -1.0)                            # 5 ReLU(1e3 s - 1)
    W1 = np.array(R); b1 = np.array(B)
    # layer 2: inputs are 6 hidden units; Ce' is index 2
    ke = knots(v['Emax'], v['EC50'], 0.003)
    kh = knots(v['HRmax'], v['EC50_hr'], 0.02)
    fE = lambda c: v['Emax']*c/(v['EC50']+c)
    fH = lambda c: v['HRmax']*c/(v['EC50_hr']+c)
    WE, bE, rE, cE = interpolant_rows(6, 2, fE, ke)
    WH, bH, rH, cH = interpolant_rows(6, 2, fH, kh)
    BIG = 1e4
    base = np.zeros((4, 6)); base[0, 0] = 1; base[1, 1] = 1; base[2, 2] = 1
    base[3, 3] = 1; base[3, 4] = -BIG; base[3, 5] = BIG
    bb = np.array([0, 0, 0, 1.0])
    W2 = np.vstack([base, WE, WH]); b2 = np.concatenate([bb, bE, bH])
    n2 = W2.shape[0]
    W3 = np.zeros((6, n2)); b3 = np.zeros(6)
    for i in range(4): W3[i, i] = 1
    W3[4, 4:4+len(rE)] = rE; b3[4] = cE
    W3[5, 4+len(rE):] = rH; b3[5] = cH
    net = ReLUNet(); net.add(W1, b1); net.add(W2, b2); net.add(W3, b3)
    return net
for name in spec['variants']:
    net = build(spec['variants'][name]); net.save(f'attempts/sonnet2/weights_{name}.npz')
    print(name, net.n_params, [w.shape for w, _ in net.layers])
