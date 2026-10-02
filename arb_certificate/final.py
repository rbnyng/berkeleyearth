import sys, json
from flint import arb, ctx
SCAN = "--scan" in sys.argv
ctx.prec = 1400 if SCAN else 1600            # set before importing kern (its constants use it)
from kern import C1, C2, C3, C4, at_pi, outpath
from bound import coeffs_and_tail, tail_norm, ineq9, least_N0

RHO, NT, KMAX = arb('0.15'), 170, 1100
if SCAN:
    RHOA, NTA, KMAXA, MF, ALLOW = arb('1e-4'), 60, 60, 1200, False
else:
    RHOA, NTA, KMAXA, MF, ALLOW = arb('0.05'), 60, 400, 6000, True

A, B, MA, MB = coeffs_and_tail(C1, C2, C3, C4, NT, RHO)
qpi, SC, Q = at_pi()
At, Gm, MAt, MGm = coeffs_and_tail(*Q, NTA, RHOA, SC)
print("M_A =", MA.str(6), " M_B =", MB.str(6), " M_At =", MAt.str(6), " M_Gm =", MGm.str(6), flush=True)
ORIGIN, ANTIPODE = (A, B, MA, MB, RHO), (At, Gm, MAt, MGm, RHOA)

PSIP = abs(Q[0])*SC
def N0_for(K, Rn):
    eps = [B[k] + Gm[k] for k in range(K)]    # B, Gm hold |beta_k|, |gamma_k|
    return least_N0(K, C1, PSIP, Rn, eps), eps

if SCAN:
    best = None
    for K in (8, 10, 12, 14, 16, 18, 20):
        for delta in ('0.04', '0.06', '0.08', '0.10', '0.12'):
            try: Rn, _ = tail_norm(K, delta, '0.01', ORIGIN, ANTIPODE, KMAX, KMAXA, MF, ALLOW)
            except AssertionError as e: print(K, delta, "skip", e); continue
            N0, _ = N0_for(K, Rn)
            print(f"K={K:3d} delta={delta}: ||R^(K)||_1 <= {float(Rn):.4e}   ->  N0 = {N0}", flush=True)
            if N0 and (best is None or N0 < best[0]): best = (N0, K, delta, Rn)
    print("\nBEST:", best[:3], " ||R||=", float(best[3]))
    json.dump({'N0': best[0], 'K': best[1], 'delta': best[2], 'Rnorm': best[3].str(12, radius=True)},
              open(outpath('opt.json'), 'w'))
    sys.exit()

K, DELTA, DP = 20, '0.06', '0.01'
Rn, pieces = tail_norm(K, DELTA, DP, ORIGIN, ANTIPODE, KMAX, KMAXA, MF, ALLOW)
for nm, v in pieces.items():
    print(f"  {nm:28s} <= {float(v):.5e}", flush=True)
print("\n||R^(K)||_{L1} <=", Rn.str(8), flush=True)

N0, EPS = N0_for(K, Rn)
assert N0 is not None and N0 <= 700, "tail threshold must lie inside the certified scan (n <= 700)"
_, l, r  = ineq9(N0,   K, C1, PSIP, Rn, EPS)
_, _, r1 = ineq9(N0-1, K, C1, PSIP, Rn, EPS)
print(f"\nN0 = {N0}   lhs {l.str(10)}  rhs {r.str(10)}  | at N0-1: rhs {r1.str(10)}")
# sanity check only: monotonicity of rhs is proved analytically
vals = [ineq9(n, K, C1, PSIP, Rn, EPS)[2] for n in range(N0, N0+400, 25)]
print("rhs decreasing on sample:", all(vals[i] > vals[i+1] for i in range(len(vals)-1)))
json.dump({'K': K, 'delta': DELTA, 'dp': DP, 'Rnorm': Rn.str(10, radius=True), 'N0': N0,
           'lhs': l.str(12, radius=True), 'rhs': r.str(12, radius=True),
           'beta':  [B[k].str(10, radius=True) for k in range(K)],
           'gamma': [Gm[k].str(10, radius=True) for k in range(K)]}, open(outpath('final.json'), 'w'))
