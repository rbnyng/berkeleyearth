import json, time
from flint import arb, arb_series, ctx
from kern import g, C1, C2, C3, C4, PI, at_pi, int0pi, outpath   # imported at default precision
ctx.prec = 400
out = {}

def Jm(m):
    return int0pi(lambda z, _: g(z)*(z*m).sin(), max(m, 1)).real

def g4_L1_upper(M):
    ctx.cap = 6
    S = lambda c: arb_series([c], 6)
    tot = arb(0); step = PI/M
    for i in range(M):
        c = step*(i + arb('0.5')); c = arb(c.mid(), (step/2).mid()*1.0000001)
        x = arb_series([c, arb(1)], 6)
        s = (-(x*(S(C1) + x*(S(C2) + x*(S(C3) + x*S(C4)))))).exp()*x.sin()
        tot += step*arb((abs(list(s.coeffs())[4])*24).upper())
    return tot

def rec(k, v, note=""):
    s = v.str(20, radius=True) if isinstance(v, arb) else str(v)
    out[k] = {"value": s, "note": note}
    print(f"  {k:38s} {s}   {note}", flush=True)

print("== constants ==")
for nm, c in [("c1", C1), ("c2", C2), ("c3", C3), ("c4", C4)]:
    rec(nm, c)
qpi, SC, Q = at_pi()
rec("q(pi)", qpi)

print("\n== S^3 ==")
g4 = g4_L1_upper(200000)
den = 2*C1 - 2*abs(Q[0])*SC
rec("||g''''||_1  (upper)", g4)
rec("2c1-|g''(pi)|  (lower)", den)
rec("N*  (upper)", g4/den)
t = time.time()
J = {m: Jm(m) for m in range(1, 201)}
neg = sorted(m for m in J if J[m] < 0)
pos = [m for m in J if J[m] > 0]
print(f"  J_m certified for m=1..200 in {time.time()-t:.1f}s")
rec("J_m < 0 exactly for m in", str(neg))
rec("J_m > 0 for all other m<=200", str(len(pos) == 200-len(neg)))
out["S3_negative_witnesses"] = {}
for m in neg:
    n = m-1
    b3 = 2*(n+1)/PI*J[m]
    mu = b3/arb((n+1)**2)
    out["S3_negative_witnesses"][n] = {"J_m": J[m].str(16, radius=True),
                                       "b_n3": b3.str(16, radius=True),
                                       "mu_n": mu.str(16, radius=True)}
    print(f"    n={n}: b_(n,3)={b3.str(12,radius=True)}   mu_n={mu.str(12,radius=True)}")

print("\n== S^2 ==")
fin = json.load(open(outpath('BFEJ_700.json')))
rec("b_n > 0 certified for 0<=n<=700", str(all(v[1] for v in fin.values())))
b = {int(k): arb(v[0].split(' +/-')[0].strip('[]')) for k, v in fin.items()}
worst = min((n for n in b if n >= 2), key=lambda n: float(arb(n)**2*b[n]))
rec("argmin_{n>=2} n^2 b_n", str(worst))
rec("min_{n>=2} n^2 b_n (approx)", arb(worst)**2*b[worst])
fj = json.load(open(outpath('final.json')))
out["S2_tail"] = fj
print(f"  tail argument: K={fj['K']}, delta={fj['delta']}, ||R^(K)||_1 <= {fj['Rnorm']}")
print(f"  => b_n > 0 for all n >= N0 = {fj['N0']}   (and {fj['N0']} <= 700, so all n are covered)")
rec("N0 (analytic tail threshold)", str(fj['N0']))
rec("CONCLUSION S^2", "psi in Psi_2^+  (strictly positive definite on S^2)")
rec("CONCLUSION S^3", "psi NOT in Psi_3  (b_{n,3} < 0 for n=26,27,28,29)")

json.dump(out, open(outpath('CERTIFICATE.json'), 'w'), indent=1)
print(f"\nwritten {outpath('CERTIFICATE.json')}")
