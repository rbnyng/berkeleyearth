import os
from functools import lru_cache
import numpy as np
from scipy.special import roots_legendre, kv, gamma

SMOKE = bool(int(os.environ.get("SMOKE", "0")))

RE = 6371.0
C1, C2, C3, C4 = 2.4541e-4*RE, 5.3881e-7*RE**2, -2.7452e-11*RE**3, 8.3007e-14*RE**4
def psi(t): return np.exp(-(C1*t + C2*t**2 + C3*t**3 + C4*t**4))

def matern(theta, nu, rho):
    t = np.atleast_1d(np.asarray(theta, float))
    arg = np.sqrt(2*nu)*t/rho
    out = np.ones_like(t); m = arg > 1e-12
    out[m] = 2**(1-nu)/gamma(nu)*arg[m]**nu*kv(nu, arg[m])
    return out

def matern_origin_const(nu, rho):
    return 2*nu*(np.sqrt(2*nu)/rho)**(2*nu)

def matern_slope_at_pi(nu, rho, h=1e-6):
    return float((matern(np.pi, nu, rho)[0] - matern(np.pi-h, nu, rho)[0])/h)

@lru_cache(maxsize=None)
def _gauss(nquad):
    x, w = roots_legendre(nquad)
    return 0.5*np.pi*(x+1), 0.5*np.pi*w

def legendre_coeffs(f, nmax, nquad=60000):
    th, wq = _gauss(nquad)
    fw = f(th)*np.sin(th)*wq; c = np.cos(th)
    b = np.empty(nmax+1); Pm = np.ones_like(c); Pc = c.copy()
    b[0] = 0.5*np.sum(fw*Pm); b[1] = 1.5*np.sum(fw*Pc)
    for n in range(1, nmax):
        Pn = ((2*n+1)*c*Pc - n*Pm)/(n+1); Pm, Pc = Pc, Pn
        b[n+1] = (2*n+3)/2*np.sum(fw*Pn)
    return b

def simpson_grid(M):
    t = np.linspace(0, np.pi, M); h = np.pi/(M-1)
    w = np.ones(M); w[1:-1:2] = 4; w[2:-1:2] = 2
    return t, h, w

def J_values(ms, M):
    t, h, w = simpson_grid(M)
    gq = psi(t)*np.sin(t)
    return np.array([h/3*np.sum(w*gq*np.sin(m*t)) for m in ms])

def sphere_points(rng, N, dim=4, dtype=np.float64):
    X = rng.standard_normal((N, dim)).astype(dtype)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X

def U(n, x):
    th = np.arccos(np.clip(x, -1, 1)); s = np.sin(th)
    return np.where(s > 1e-9, np.sin((n+1)*th)/np.where(s > 1e-9, s, 1), (n+1)*np.sign(x)**n)
