import sys, time, json
from flint import arb, ctx
from kern import g, int0pi, bn, outpath      # imported at default precision (as in the paper's run)

def Cm(m):
    return int0pi(lambda z, _: g(z)*(z*m).cos(), max(1, m)).real

def run(NMAX, prec=400):
    ctx.prec = prec
    t = time.time()
    C = [Cm(m) for m in range(NMAX+1)]
    print(f"  C_m for m<={NMAX} in {time.time()-t:.1f}s   C_0={C[0].str(10)}  C_{NMAX}={C[NMAX].str(6)}",
          flush=True)
    G = [arb(1)]
    for k in range(1, NMAX+1):
        G.append(G[-1]*arb(2*k-1)/arb(2*k))
    b = []
    for n in range(NMAX+1):
        s = arb(0)
        for k in range(n+1):
            s += G[k]*G[n-k]*C[abs(n-2*k)]
        b.append(arb(2*n+1)/2*s)
    return b

def check(b):
    ctx.prec = 300
    for n in (0, 5, 26, 27, 100, 200, 300):
        d = bn(n)
        ov = d.overlaps(b[n])
        print(f"   n={n:4d} Fejer {b[n].str(12,radius=True):>34}  direct {d.str(12,radius=True):>34}  overlap={ov}")
        assert ov, f"Fejer and direct enclosures disagree at n={n}"

if __name__ == "__main__":
    NMAX = int(sys.argv[1]); prec = int(sys.argv[2]) if len(sys.argv) > 2 else 400
    b = run(NMAX, prec)
    bad = [n for n in range(NMAX+1) if not b[n] > 0]
    print(f"\n  certified b_n > 0 for all 0<=n<={NMAX}: {not bad}   (failures: {bad[:10]})", flush=True)
    worst = min(range(NMAX+1), key=lambda n: float(arb(n)**2*b[n]) if n else 1e9)
    print(f"  min over n of n^2*b_n : n={worst}, n^2 b_n = {(arb(worst)**2*b[worst]).str(10)}")
    print(f"  max relative ball radius: {max(float(b[n].rad()/abs(b[n])) for n in range(1,NMAX+1)):.2e}")
    print("\n  cross-check vs direct Legendre quadrature:")
    check(b)
    json.dump({str(n): [b[n].str(20, radius=True), bool(b[n] > 0)] for n in range(NMAX+1)},
              open(outpath(f'BFEJ_{NMAX}.json'), 'w'))
