import sys, time
import numpy as np
from numpy.polynomial.legendre import leggauss
from math import pi
import mpmath as mp
from common import SMOKE, C1, C2, C3, C4, psi, U, sphere_points

def psi_so3(t):   # SO(3) = S^3/{+-1}
    return psi(np.minimum(t, pi - t))

def hopf_grid_quadform(n_eta, n1, n2, deg=27, kernel=psi):
    t, glw = leggauss(n_eta)
    eta = 0.5*np.arccos(t); om = glw/np.sum(glw)
    ce, se = np.cos(eta), np.sin(eta)
    d1 = np.cos(2*pi*np.arange(n1)/n1); d2 = np.cos(2*pi*np.arange(n2)/n2)
    g = U(deg, ce[:, None]*d1[None, :])
    g2 = float(np.sum(om[:, None]*g**2)/n1)
    Q = 0.0
    for a in range(n_eta):
        ip = (ce[a]*ce)[:, None, None]*d1[None, :, None] + (se[a]*se)[:, None, None]*d2[None, None, :]
        P = kernel(np.arccos(np.clip(ip, -1, 1))).mean(axis=2)
        for b in range(n_eta):
            # circulant action (C g_b)[i] = sum_{i'} P[b, (i-i') mod n1] g[b, i'] by FFT
            Cg = np.fft.irfft(np.fft.rfft(P[b])*np.fft.rfft(g[b]), n=n1)
            Q += om[a]*om[b]*float(g[a] @ Cg)/(n1*n1)
    return Q, g2, n_eta*n1*n2

def random_design_quadform(N, deg, seed, kernel=psi, block=500):
    X = sphere_points(np.random.default_rng(seed), N)
    g = U(deg, X[:, 0]); w = g/N
    Q = 0.0
    for i0 in range(0, N, block):
        ip = X[i0:i0+block] @ X.T
        np.clip(ip, -1, 1, out=ip); np.arccos(ip, out=ip)
        Q += float(w[i0:i0+block] @ (kernel(ip) @ w))
    return Q, float(np.mean(g**2))

def gp_regression(N=4000, Ntest=1000, noise=0.05, seed=7):
    rng = np.random.default_rng(seed)
    X = sphere_points(rng, N+Ntest)
    Uc = sphere_points(rng, 6)
    al = rng.standard_normal(6)
    f = sum(a*np.tanh(2*X @ u) for a, u in zip(al, Uc))
    y = f + noise*rng.standard_normal(N+Ntest)
    Xtr, Xte, ytr, fte = X[:N], X[N:], y[:N], f[N:]
    TH = np.arccos(np.clip(Xtr @ Xtr.T, -1, 1)); np.fill_diagonal(TH, 0.0)
    K = psi(TH); Kn = K + noise**2*np.eye(N)
    try:
        np.linalg.cholesky(Kn); chol_ok = True
    except np.linalg.LinAlgError:
        chol_ok = False
    Ks = psi(np.arccos(np.clip(Xte @ Xtr.T, -1, 1)))
    mu = Ks @ np.linalg.solve(Kn, ytr)
    var = 1.0 - np.einsum('ij,ji->i', Ks, np.linalg.solve(Kn, Ks.T))
    return dict(lam_min=float(np.linalg.eigvalsh(K)[0]), chol_ok=chol_ok,
                rmse=float(np.sqrt(np.mean((mu-fte)**2))), base=float(np.sqrt(np.mean(fte**2))),
                min_predvar=float(var.min()), neg_predvars=int((var < 0).sum()))

N_RANDOM = 8000 if SMOKE else 80000

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "net"):
        grids = [(20, 64, 64, 10), (20, 64, 64, 27)] + ([] if SMOKE else [(34, 96, 96, 27), (28, 80, 80, 27)])
        for (ne, n1, n2, deg) in grids:
            t0 = time.time()
            Q, g2, N = hopf_grid_quadform(ne, n1, n2, deg)
            print(f"deg={deg:2d} grid=({ne},{n1},{n2}) N={N:7d}: w^T K w = {Q: .4e}  "
                  f"(<g^2>={g2:.6f})  [{time.time()-t0:.1f}s]", flush=True)
        mp.mp.dps = 30
        for n in (10, 27):
            J = mp.quad(lambda th: mp.e**(-(C1*th+C2*th**2+C3*th**3+C4*th**4))*mp.sin(th)*mp.sin((n+1)*th),
                        mp.linspace(0, mp.pi, 2*n+4))
            print(f"   reference mu_{n} = {mp.nstr(2*(n+1)/mp.pi*J/(n+1)**2, 6)}")
    if which in ("all", "random"):
        for seed in (1, 2, 3):
            t0 = time.time()
            Q, g2 = random_design_quadform(N_RANDOM, 27, seed)
            print(f"seed={seed}: random design N={N_RANDOM}: w^T K w = {Q:.4e}  E[g^2]={g2:.4f}  "
                  f"[{time.time()-t0:.0f}s]  (positive: diagonal wall)", flush=True)
    if which in ("all", "so3"):
        deg = 28
        grids = [(16, 56, 56), (18, 60, 60), (20, 64, 64)] + ([] if SMOKE else [(24, 72, 72), (28, 80, 80), (34, 96, 96)])
        for ne, n1, n2 in grids:
            t0 = time.time(); Q, _, N = hopf_grid_quadform(ne, n1, n2, deg, kernel=psi_so3)
            print(f"SO3 deg={deg} grid=({ne},{n1},{n2}) N={N:7d}: w^T K w = {Q: .4e}  [{time.time()-t0:.0f}s]", flush=True)
        Q, _ = random_design_quadform(N_RANDOM, deg, 1, kernel=psi_so3)
        print(f"SO3 random design deg={deg} N={N_RANDOM}: w^T K w = {Q: .4e}", flush=True)
    if which in ("all", "gp"):
        t0 = time.time()
        r = gp_regression(N=1000 if SMOKE else 4000)
        print(f"GP N={1000 if SMOKE else 4000}: lam_min(K)={r['lam_min']:.3e}  chol_ok={r['chol_ok']}  "
              f"rmse={r['rmse']:.4f} (signal {r['base']:.3f})  min predictive var={r['min_predvar']:.3e}  "
              f"#negative={r['neg_predvars']}  [{time.time()-t0:.0f}s]")
