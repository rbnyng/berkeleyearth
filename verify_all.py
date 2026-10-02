import numpy as np, mpmath as mp, sympy as sp
from math import pi, comb
from scipy.integrate import quad
from common import SMOKE, C1 as c1, C2 as c2, C3 as c3, C4 as c4, psi, legendre_coeffs, simpson_grid, J_values

print(f"[1] c = ({c1:.8f}, {c2:.8f}, {c3:.8f}, {c4:.8f})")

NMAX = 400 if SMOKE else 3000
b60 = legendre_coeffs(psi, NMAX, 20000 if SMOKE else 60000)
b90 = legendre_coeffs(psi, 400 if SMOKE else 500, 30000 if SMOKE else 90000)
assert b60.min() > 0 and b90.min() > 0
print(f"[2] min b_n (n<=3000): {b60.min():.3e}; cross-resolution rel diff "
      f"{np.max(np.abs(b60[:len(b90)]-b90)/np.abs(b90)):.1e}")
for th0 in (0.1, 0.3):
    x = np.cos(th0); P = np.zeros(NMAX+1); P[0], P[1] = 1, x
    for n in range(1, NMAX): P[n+1] = ((2*n+1)*x*P[n] - n*P[n-1])/(n+1)
    assert abs(np.dot(b60, P) - psi(th0)) < (2e-3 if SMOKE else 1e-5)
print(f"    reconstruction OK; n^2 b_n at n={NMAX}: {NMAX**2*b60[NMAX]:.4f} (c1={c1:.4f})")

tg = np.linspace(0, pi, 60001 if SMOKE else 400001)
T = sp.symbols('t', positive=True)
g_sym = sp.exp(-(c1*T + c2*T**2 + c3*T**3 + c4*T**4))*sp.sin(T)
gsups = 10*np.array([np.max(np.abs(sp.lambdify(T, sp.diff(g_sym, T, k), 'numpy')(tg)))
                     for k in range(5)])
NN = 60 if SMOKE else 260
B4 = lambda n: sum(comb(4, j)*gsups[4-j]*max(n, 1)**j for j in range(5))
err_target = 0.1*1.56*2/((2*NN+1)*NN**2)
h = (err_target*180/(pi*B4(NN)))**0.25
Nint = int(np.ceil(pi/h/2))*2
tE, hE, wE = simpson_grid(Nint+1)
gE = psi(tE)*np.sin(tE); xE = np.cos(tE)
Pm = np.ones_like(xE); Pc = xE.copy(); ok, worst = True, np.inf
for n in range(NN+1):
    if n >= 2:
        Pn = ((2*n-1)*xE*Pc - (n-1)*Pm)/n; Pm, Pc = Pc, Pn
    I = hE/3*np.sum(wE*gE*(Pm if n == 0 else Pc))
    marg = (2*n+1)/2*(I - (pi/180)*hE**4*B4(n) - Nint*2e-16*0.08)
    worst = min(worst, marg); ok &= marg > 0
assert ok
print(f"[3] floating-point Simpson check with a priori error term (not interval-certified; see arb_certificate/): "
      f"b_n>0 for n<=260 ({Nint} intervals); worst margin {worst:.2e}")

ms = np.arange(2, 147)
J = J_values(ms, 60001 if SMOKE else 200001)
negs = {int(m): v for m, v in zip(ms, J) if v < 0}
assert set(negs) == {27, 28, 29, 30}
mp.mp.dps = 40
J29 = mp.quad(lambda t: mp.e**(-(c1*t+c2*t**2+c3*t**3+c4*t**4))*mp.sin(t)*mp.sin(29*t),
              [0, mp.mpf(1)/4, 1, mp.pi])
assert J29 < 0
g4 = sp.lambdify(T, sp.diff(g_sym, T, 4), 'numpy')
g4n = sum(quad(lambda t: abs(g4(t)), a, b, limit=400)[0]
          for a, b in [(0, .05), (.05, .3), (.3, 1.2), (1.2, pi)])
Nstar = g4n/(2*c1)
print(f"[4] J_m<0 exactly at m=27..30 (worst {min(negs.values()):.3e}; "
      f"40-digit J_29 = {mp.nstr(J29,6)}); N* = {Nstar:.0f}")
print(f"    b_27,3 = {2*28/pi*negs[28]:.3e}; mu_27 = {2*28/pi*negs[28]/28**2:.3e}")
