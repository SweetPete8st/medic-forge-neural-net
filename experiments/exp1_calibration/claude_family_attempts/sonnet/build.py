import math, sys, yaml, numpy as np
sys.path.insert(0, '..'); sys.path.insert(0, '../..')
from wsp_layers import ReLUNet, interpolant_rows
spec = yaml.safe_load(open('../../pk_spec_dev.yaml'))
LN2 = math.log(2)
def knots_for(ec, cmax, tol):
    # greedy: step h with h^2/8*|f''(x)| = tol (f'' decreasing in x so safe); f(c)=c/(ec+c)
    ks=[0.0]; x=0.0
    while x<cmax:
        f2=2*ec/(ec+x)**3
        x=x+math.sqrt(8*tol/f2); ks.append(x)
    return ks
for name, v in spec['variants'].items():
    k=lambda h: LN2/(h*60)
    ea=math.exp(-k(v['ka_half_min'])); ek=math.exp(-k(v['ke_half_min'])); eo=math.exp(-k(v['keo_half_min']))
    ks=sorted(set(knots_for(v['EC50'],6.0,0.002))|set(knots_for(v['EC50_hr'],6.0,0.002)))
    ks=[round(x,12) for x in ks]
    fE=lambda c: v['Emax']*c/(v['EC50']+c); fH=lambda c: v['HRmax']*c/(v['EC50_hr']+c)
    K=len(ks)-1
    # inputs: 0 Ad,1 Ac,2 Ce,3 tau,4 ud,5 uc
    W1=np.zeros((6,6)); b1=np.zeros(6); S=1000.0
    W1[0,[0,4]]=ea
    W1[1,[0,4]]=ek*(1-ea); W1[1,[1,5]]=ek
    W1[2,[0,4]]=(1-eo)*ek*(1-ea); W1[2,[1,5]]=(1-eo)*ek; W1[2,2]=eo
    W1[3,3]=1.0
    W1[4,[4,5]]=S               # g1 = ReLU(S*u)
    W1[5,[4,5]]=S; b1[5]=-1.0   # g2 = ReLU(S*u-1); gate=g1-g2
    big=1e4
    n2=4+K
    W2=np.zeros((n2,6)); b2=np.zeros(n2)
    W2[0,0]=1; W2[1,1]=1; W2[2,2]=1
    W2[3,3]=1; W2[3,4]=-big; W2[3,5]=big; b2[3]=1.0   # tau'=ReLU(tau+1-big*gate)
    Wi,bi,ro,bias=interpolant_rows(6,2,fE,ks); _,_,roH,biasH=interpolant_rows(6,2,fH,ks)
    for j in range(K): W2[4+j,2]=1; b2[4+j]=bi[j]
    W3=np.zeros((6,n2)); b3=np.zeros(6)
    for i in range(4): W3[i,i]=1
    W3[4,4:]=ro; b3[4]=bias; W3[5,4:]=roH; b3[5]=biasH
    net=ReLUNet(); net.add(W1,b1); net.add(W2,b2); net.add(W3,b3)
    net.save(f'weights_{name}.npz')
    print(name,'knots',K,'params',net.n_params)
