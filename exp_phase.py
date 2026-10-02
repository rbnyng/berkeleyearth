import numpy as np
from common import SMOKE, C1, C2, C3, C4, simpson_grid

t, h, ww = simpson_grid(6001 if SMOKE else 20001)
sint = np.sin(t); x = np.cos(t)
mmax = nmax = 120 if SMOKE else 250

S = np.sin(np.outer(np.arange(2, mmax+1), t))
P = np.empty((nmax+1, len(t))); P[0] = 1; P[1] = x
for n in range(1, nmax):
    P[n+1] = ((2*n+1)*x*P[n] - n*P[n-1])/(n+1)
wS = ww*h/3
scale = (2*np.arange(nmax+1) + 1)/2

npts = 21 if SMOKE else 61
c3s = np.linspace(-16, 2, npts)
c4s = np.linspace(40, 240, npts)
phase = np.zeros((len(c4s), len(c3s)), dtype=int)
LABELS = {0: "improper", 1: "invalid on S^2", 2: "valid on S^2 only", 3: "valid on both"}

for i, c4 in enumerate(c4s):
    psis = np.exp(-(C1*t + C2*t**2 + np.outer(c3s, t**3) + c4*t**4))
    proper = (psis.max(axis=1) <= 1 + 1e-9) & (psis[:, -1] < 1e-3)
    gmat = psis*sint*wS
    minJ = (gmat @ S.T).min(axis=1)
    bmin = (gmat @ P.T*scale).min(axis=1)
    phase[i] = np.select([~proper, bmin <= 0, minJ <= 0], [0, 1, 2], default=3)

np.save("phase.npy", phase)
np.save("phase_axes.npy", np.array([c3s, c4s], dtype=object), allow_pickle=True)

i = np.argmin(abs(c4s - C4)); j = np.argmin(abs(c3s - C3))
vals, counts = np.unique(phase, return_counts=True)
print("phase cell counts:", {LABELS[int(v)]: int(cnt) for v, cnt in zip(vals, counts)})
print(f"fitted model (c3={C3:.3f}, c4={C4:.3f}) -> class {phase[i,j]} ({LABELS[phase[i,j]]})")
assert phase[i, j] == 2, "fitted model should be valid on S^2 only"
print("DONE")
