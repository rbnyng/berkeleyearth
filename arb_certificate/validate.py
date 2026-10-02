import json
from flint import arb, acb, ctx
ctx.prec = 1200                               # set before importing kern (its constants use it)
from kern import C1, C2, C3, C4, int0pi, bn, main_term, outpath
from bound import AB, bk, Phi

print("[1] closed form  b_n[(1-u)^{k+1/2}]  vs rigorous quadrature")
for (n, k) in [(3, 0), (10, 0), (25, 2), (40, 5)]:
    s = arb(2*k+1)/2
    f = lambda z, _: (1-z.cos())**acb(s)*acb.legendre_p(z.cos(), n)*z.sin()
    I = (acb(2*n+1)/2*int0pi(f, max(n, 1))).real
    print(f"   n={n:3d} k={k}:  closed {bk(n,k).str(14)}   quad {I.str(14)}")

print("\n[2] main term identity:  beta_0*B_n^-(0) == c1/((n+3/2)(n-1/2))")
ctx.cap = 25
A, B = AB(C1, C2, C3, C4, 25)
for n in (100, 700):
    print(f"   n={n}:  {(B[0]*bk(n,0)).str(16)}   vs   {main_term(n).str(16)}")

fj = json.load(open(outpath('final.json')))
K = fj['K']; Rn = arb(arb(fj['Rnorm']).upper())
print(f"\n[3] residual |b_n[R]| vs the certified bound  Phi_K(n)*||R^(K)||_1   (K={K}, ||R|| <= {float(Rn):.4e})")
for n in (300, 500, 700, 900):
    b = bn(n)
    R = b - sum(B[k]*bk(n, k) for k in range(K))
    bound = Phi(n, K)*Rn
    print(f"   n={n}:  |b_n[R]| = {float(abs(R)):.4e}      bound = {float(bound):.4e}   ok={bool(abs(R)<bound)}")
    print(f"            b_n = {b.str(12)}   main = {main_term(n).str(12)}")
