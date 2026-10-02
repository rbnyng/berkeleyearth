# code

Run with `./run_all.sh` (Python ≥ 3.10, packages pinned in requirements.txt)

| Script | Purpose | Paper |
|---|---|---|
| `verify_all.py` | floating-point checks of the main numbers | Sec.5 |
| `exp_invisibility.py` | Gram-matrix $\lambda_{\min}$ on $\mathbb S^3$ (float32) and Rayleigh–Ritz | Sec.6.1, Fig. 3a |
| `exp_phase.py` | Floating-point phase diagram over $(c_3,c_4)$ | Sec.6.3, Fig. 3b |
| `exp_matern.py` | Geodesic Matern calibration and Proposition 1 | Sec.6.4, Fig. 4, Table 3 |
| `exp_networks.py [net\|random\|so3\|gp]` | Hopf-grid and random-design network averages on $\mathbb S^3$ and on $SO(3)$ (kernel $\psi(\min(\theta,\pi-\theta))$); GP regression | Sec.6.2, Table 2 |
| `make_figures.py` | Figures | |
| `common.py` | helpers. the model in floating point, Gauss–Legendre $b_{n,2}$, Simpson $J_m$, Matérn, points on $\mathbb S^3$ | |
