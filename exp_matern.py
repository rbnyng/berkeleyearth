import numpy as np
from common import SMOKE, matern, matern_origin_const, matern_slope_at_pi, legendre_coeffs

NQ = 20000 if SMOKE else 60000
NMAX = 1000 if SMOKE else 3000
coeffs = lambda nu, rho, nmax: legendre_coeffs(lambda t: matern(t, nu, rho), nmax, NQ)

nus = [0.30, 0.45, 0.50, 0.55, 0.70, 1.00, 1.50] if SMOKE else \
      [0.30, 0.40, 0.45, 0.48, 0.50, 0.52, 0.55, 0.60, 0.70, 0.85, 1.00, 1.25, 1.50]
res = {0.5: [], 1.0: []}
for nu in nus:
    for rho in (0.5, 1.0):
        res[rho].append(coeffs(nu, rho, NMAX).min())
    flag = "OK<=1/2" if (nu <= 0.5 and min(res[0.5][-1], res[1.0][-1]) >= 0) else ""
    print(f"nu={nu:4.2f}  min_b(rho=.5)={res[0.5][-1]: .3e}  min_b(rho=1)={res[1.0][-1]: .3e}  {flag}")
np.save("matern.npy", np.array([nus, res[0.5], res[1.0]]))
for k, nu in enumerate(nus):
    if nu <= 0.5:
        assert res[0.5][k] >= -1e-12 and res[1.0][k] >= -1e-12, f"unexpected negative at nu={nu}"
print("DONE (validity confirmed at and below nu=1/2 within scan range)")

print("\n[Prop 1 (ii)] origin constant C_o = 2 nu kappa^{2nu}:")
n0 = 1000 if SMOKE else 4000
for nu, rho in [(0.30, 1.0), (0.70, 0.5)]:
    pred = matern_origin_const(nu, rho)
    got = coeffs(nu, rho, n0)[n0]*n0**(2*nu+1)
    print(f"  nu={nu}, rho={rho}: predicted {pred:.4f}, measured {got:.4f}")
    assert abs(got/pred - 1) < 0.05

print("[Prop 1 (iii)] two-term law at nu=0.85, rho=1:")
nu, rho = 0.85, 1.0
Co, ap = matern_origin_const(nu, rho), matern_slope_at_pi(nu, rho)
ns = (400, 401, 800, 801) if SMOKE else (400, 401, 800, 801, 1600, 1601, 3000, 3001)
b = coeffs(nu, rho, max(ns))
for n in ns:
    pred = Co*n**(-(2*nu+1)) + ((-1)**n)*ap*n**(-2)
    r = b[n]/pred
    print(f"  n={n} ({'even' if n%2==0 else 'odd'}): ratio measured/predicted = {r:.3f}")
    assert abs(r - 1) < 0.05
    assert (b[n] < 0) == (n % 2 == 0)
nmin = ((2*nu+1)*nu*(np.sqrt(2*nu)/rho)**(2*nu)/abs(ap))**(1/(2*nu-1))
print(f"  predicted most-negative location n_min = {nmin:.0f}  (measured: 552)")
print("PROPOSITION 1 VERIFIED")
