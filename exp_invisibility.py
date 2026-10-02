import numpy as np
from scipy.special import eval_chebyu
from scipy.linalg import eigh
from common import SMOKE, psi, J_values, sphere_points

def gram(X):
    # float32 as in App. D; backward error ~1e-7*||K|| is far below the reported values
    return psi(np.arccos(np.clip(X @ X.T, -1.0, 1.0))).astype(np.float32)

Ns = [500, 1000, 2000] if SMOKE else [2000, 4000, 6000, 8000, 10000]
floors = []
for i, N in enumerate(Ns):
    X = sphere_points(np.random.default_rng(1000 + i), N, dtype=np.float32)
    lm = float(np.linalg.eigvalsh(gram(X))[0])
    floors.append((N, lm))
    print(f"[A] S^3  N={N:6d}: lambda_min = {lm: .4e}  (positive: {lm > 0})")
floors = np.array(floors)
np.save("floors.npy", floors)
assert (floors[:, 1] > 0).all(), "empirical floors unexpectedly non-positive"

J28, = J_values([28], 60001 if SMOKE else 200001)
mu_min = 2*28/np.pi*J28/28**2
print(f"    certified mu_min (deg 27) = {mu_min:.3e};  1/|mu_min| = {1/abs(mu_min):.2e}")

def ritz_deg27(N, P, seed=7):
    rng = np.random.default_rng(seed)
    X = sphere_points(rng, N, dtype=np.float32)
    K = gram(X)
    poles = sphere_points(rng, P)
    Phi = eval_chebyu(27, (X @ poles.T.astype(np.float32)))
    G = Phi.T @ Phi; w, V = eigh(G.astype(np.float64))
    keep = w > 1e-8*w.max(); W = V[:, keep]/np.sqrt(w[keep])
    C = Phi.T @ (K @ Phi)
    return float(np.linalg.eigvalsh(W.T @ C.astype(np.float64) @ W)[0])

Nr = 2000 if SMOKE else 10000
r = ritz_deg27(Nr, P=200 if SMOKE else 400)
print(f"[B] Rayleigh-Ritz min over deg-27 subspace (N={Nr}): {r:.4f}")
print(f"    (positive => defect undetectable at this N; needs N > 1/|mu_min| ~ {1/abs(mu_min):.1e})")
print("\nDONE")
