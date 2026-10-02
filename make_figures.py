import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, LogLocator
from scipy.special import sph_harm_y
from common import (RE, C1, C3, C4, psi, legendre_coeffs, J_values, matern,
                    matern_origin_const, matern_slope_at_pi)

PALETTE = ['#1f4e79', '#c0504d', '#4f81bd', '#e8a33d']
plt.rcParams.update({'font.family': 'serif', 'font.size': 13,
                     'axes.labelsize': 13, 'xtick.labelsize': 12, 'ytick.labelsize': 12,
                     'legend.fontsize': 12, 'axes.linewidth': 1.1,
                     'xtick.major.width': 1.1, 'ytick.major.width': 1.1,
                     'axes.spines.top': False, 'axes.spines.right': False})

b = legendre_coeffs(psi, 3000, 60000)

def fig_theorem():
    # drawn at its final printed size: \textwidth = 7.00 in
    fig, axs = plt.subplots(1, 3, figsize=(12.0, 4.34))
    TT = dict(loc="left", fontweight="bold", fontsize=13.5, pad=8)

    dd = np.linspace(0, 4500, 1200)
    axs[0].plot(dd, psi(dd/RE), color=PALETTE[0], lw=2.6)
    axs[0].axvspan(1000, 2000, color=PALETTE[3], alpha=0.14)
    axs[0].set_xlabel("great-circle distance (km)"); axs[0].set_ylabel("correlation")
    axs[0].set_title("(a)  The Berkeley Earth model", **TT)
    axs[0].xaxis.set_major_locator(MaxNLocator(4))

    n = np.arange(1, 3001)
    axs[1].loglog(n, b[1:3001], color=PALETTE[1], lw=2.2, label="$b_{n,2}$", zorder=3)
    m = n[24:]
    axs[1].loglog(m, C1/((m+1.5)*(m-0.5)), '--', color='0.35', lw=1.7, zorder=4,
                  label=r"$c_1/[(n{+}\frac{3}{2})(n{-}\frac{1}{2})]$")
    axs[1].axvspan(20, 50, color=PALETTE[3], alpha=0.14)
    axs[1].set_xlabel("degree $n$"); axs[1].set_ylabel("coefficient")
    axs[1].set_title(r"(b)  All coefficients positive on $\mathbb{S}^2$", **TT)
    axs[1].set_ylim(3e-8, 3e-1)
    axs[1].yaxis.set_major_locator(LogLocator(base=10, numticks=5))
    axs[1].legend(frameon=False, loc="lower left", handlelength=1.8, borderpad=0.2)

    ms = np.arange(2, 61)
    y3 = J_values(ms, 50001)*ms**3
    axs[2].plot(ms, y3, color=PALETTE[2], lw=2.3)
    axs[2].axhline(0, color='black', lw=1.1)
    axs[2].fill_between(ms, y3, 0, where=(y3 < 0), interpolate=True, color='#c23b22', lw=0, zorder=3)
    axs[2].annotate(r"$J_m<0$, $m=27$–$30$", xy=(29.3, -0.85), xytext=(35.0, 13.0),
                    fontsize=12, color='#c23b22', ha='left', va='center',
                    arrowprops=dict(arrowstyle='-|>', color='#c23b22', lw=1.3,
                                    relpos=(0.0, 0.0), shrinkA=2, shrinkB=1))
    axs[2].set_ylim(-2.6, 24.5)
    axs[2].set_xlabel("frequency $m$"); axs[2].set_ylabel("$m^3 J_m$")
    axs[2].set_title(r"(c)  Criterion crosses zero on $\mathbb{S}^3$", **TT)

    # explicit margins: no tight-bbox crop, so the LaTeX scale is exactly 7/12
    fig.subplots_adjust(left=0.065, right=0.988, bottom=0.177, top=0.889, wspace=0.243)
    fig.savefig('theorem.pdf')
    W, H = fig.get_size_inches(); sc = 7.0/W; p = axs[0].get_position()
    print('printed figure %.2f x %.2f in ; each axes %.2f w x %.2f h in (area %.2f)'
          % (W*sc, H*sc, p.width*W*sc, p.height*H*sc, p.width*W*sc*p.height*H*sc))
    plt.close(fig)

def fig_planet():
    rng = np.random.default_rng(1815)
    lat = np.linspace(-np.pi/2, np.pi/2, 181); lon = np.linspace(-np.pi, np.pi, 361)
    LON, LAT = np.meshgrid(lon, lat); colat = np.pi/2 - LAT
    T = np.zeros_like(LON)
    for n in range(1, 81):
        Cn = 4*np.pi*b[n]/(2*n+1)
        if Cn <= 0: continue
        sd = np.sqrt(Cn)
        T += sd*rng.standard_normal()*sph_harm_y(n, 0, colat, LON).real
        for m in range(1, n+1):
            Y = sph_harm_y(n, m, colat, LON)
            ar, ai = rng.standard_normal(2)*sd/np.sqrt(2)
            T += 2*(ar*Y.real - ai*Y.imag)
    T /= T.std()
    def ortho(lo, la, lo0=0.35, la0=0.45):
        cc = np.sin(la0)*np.sin(la) + np.cos(la0)*np.cos(la)*np.cos(lo-lo0)
        X = np.cos(la)*np.sin(lo-lo0)
        Y = np.cos(la0)*np.sin(la) - np.sin(la0)*np.cos(la)*np.cos(lo-lo0)
        return X, Y, cc
    X, Y, cc = ortho(LON, LAT); mask = cc <= 0.03
    msk = lambda F: np.where(mask, np.nan, F)
    def grat(a):
        for lo in np.arange(-180, 180, 30)*np.pi/180:
            la = np.linspace(-np.pi/2, np.pi/2, 200); xx, yy, c2 = ortho(np.full_like(la, lo), la)
            xx[c2 <= 0.03] = np.nan; a.plot(xx, yy, color='0.55', lw=0.35, alpha=0.7)
        for la0 in np.arange(-60, 90, 30)*np.pi/180:
            lo = np.linspace(-np.pi, np.pi, 400); xx, yy, c2 = ortho(lo, np.full_like(lo, la0))
            xx[c2 <= 0.03] = np.nan; a.plot(xx, yy, color='0.55', lw=0.35, alpha=0.7)
        a.add_patch(plt.Circle((0, 0), 1, fill=False, color='0.3', lw=1.3))
        a.set_aspect('equal'); a.axis('off'); a.set_xlim(-1.06, 1.06); a.set_ylim(-1.06, 1.06)
    def panel(a, F, title, cbar, extra=lambda a: None, **kw):
        p = a.pcolormesh(X, Y, msk(F), shading='gouraud', rasterized=True, **kw)
        extra(a); grat(a); a.set_title(title, loc="left", fontweight="bold", fontsize=12)
        fig.colorbar(p, ax=a, shrink=0.72, pad=0.02, label=cbar)

    fig, ax = plt.subplots(1, 3, figsize=(15, 5.4))
    panel(ax[0], T, "A planet drawn from the model", "temperature anomaly (s.d.)",
          cmap='RdBu_r', vmin=-2.7, vmax=2.7)
    pla, plo = 0.62, 0.05
    gc = np.arccos(np.clip(np.sin(pla)*np.sin(LAT) + np.cos(pla)*np.cos(LAT)*np.cos(LON-plo), -1, 1))
    def footprint(a):
        csl = a.contour(X, Y, msk(gc*RE), levels=[1000, 2000, 3000], colors='white', linewidths=0.7, linestyles='--')
        a.clabel(csl, fmt=lambda v: f"{v:.0f} km", fontsize=7)
        px, py, _ = ortho(np.array([plo]), np.array([pla]))
        a.plot(px, py, marker='*', ms=13, color='cyan', mec='black', mew=0.7)
    panel(ax[1], psi(gc), "What one thermometer sees", "correlation", footprint, cmap='magma', vmin=0, vmax=0.9)
    M28 = sph_harm_y(27, 14, colat, LON).real; M28 /= np.abs(M28[~mask]).max()
    panel(ax[2], M28, "The mode at the edge of validity", "degree-27 harmonic (arb.)",
          cmap='PuOr_r', vmin=-1, vmax=1)
    fig.tight_layout(); fig.savefig('planet.pdf', dpi=300, bbox_inches='tight'); plt.close(fig)

def fig_experiments():
    floors = np.load('floors.npy')
    phase = np.load('phase.npy'); c3s, c4s = np.load('phase_axes.npy', allow_pickle=True)
    mu = 7.744e-7                                          # |mu_27| (certificate.py)
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    Ns = np.geomspace(1e3, 3e6, 200)
    floor = lambda N: 4*C1/(np.pi*(3*N)**(1/3.))           # unresolved spectral mass, Sec. 6.1 / App. D
    Nstar = (4*C1/(np.pi*mu*3**(1/3.)))**0.75              # N|mu| = floor(N)  ->  ~4.9e4
    ax[0].loglog(floors[:, 0], floors[:, 1], 'o', color=PALETTE[0], ms=7, label=r"measured $\lambda_{\min}$ of Gram matrix")
    ax[0].loglog(Ns, mu*Ns, '-', color=PALETTE[2], lw=1.8, label=r"defect magnitude $N|\mu_{\min}|$")
    ax[0].loglog(Ns, floor(Ns), '--', color='gray', lw=1.2, label=r"unresolved-mass floor $\propto N^{-1/3}$")
    ax[0].axvline(Nstar, color='black', lw=0.8, ls=':', label=r"detection threshold $N^{\dagger}$")
    ax[0].set_xlabel(r"number of points $N$ on $\mathbb{S}^3$"); ax[0].set_ylabel("eigenvalue scale")
    ax[0].set_title("Empirical undetectability of the defect", loc="left", fontweight="bold")
    ax[0].legend(frameon=False, loc="lower right", fontsize=8)
    cmap = ListedColormap(['#bbbbbb', '#c23b22', '#e8a33d', '#3d7d4f'])
    ax[1].pcolormesh(c3s, c4s, phase, cmap=cmap, vmin=-0.5, vmax=3.5, shading='auto', rasterized=True)
    ax[1].plot([C3], [C4], marker='*', ms=16, color='white', mec='black', mew=1.2)
    H = [Patch(fc='#c23b22', label=r"invalid on $\mathbb{S}^2$"), Patch(fc='#e8a33d', label=r"valid on $\mathbb{S}^2$ only"),
         Patch(fc='#3d7d4f', label=r"valid on $\mathbb{S}^2$ and $\mathbb{S}^3$"),
         plt.Line2D([], [], marker='*', ls='', ms=12, mfc='white', mec='black', label="Berkeley Earth fit")]
    ax[1].legend(handles=H, loc="upper left", fontsize=8, framealpha=0.95)
    ax[1].set_xlabel("$c_3$"); ax[1].set_ylabel("$c_4$")
    ax[1].set_title(r"Validity phase diagram ($c_1, c_2$ at fit)", loc="left", fontweight="bold")
    for k, a in enumerate(ax): a.text(-0.08, 1.02, "(%s)" % "ab"[k], transform=a.transAxes, fontweight="bold")
    fig.tight_layout(); fig.savefig('experiments.pdf', dpi=300, bbox_inches='tight'); plt.close(fig)

def fig_matern():
    nus_s, m05, m10 = np.load('matern.npy')
    nu0, rho0 = 0.85, 1.0
    bm = legendre_coeffs(lambda t: matern(t, nu0, rho0), 3001, 60000)
    grid = np.unique(np.geomspace(60, 3000, 40).astype(int)//2*2)
    ns = np.union1d(grid, grid+1); vals = bm[ns]
    Co0, ap0 = matern_origin_const(nu0, rho0), matern_slope_at_pi(nu0, rho0)
    nmin = ((2*nu0+1)*nu0*(np.sqrt(2*nu0)/rho0)**(2*nu0)/abs(ap0))**(1/(2*nu0-1))
    def pred_min(nu, rho, NMAX=3000):
        n = np.arange(2, NMAX+1)
        return (matern_origin_const(nu, rho)*n**(-(2*nu+1)) + ((-1.0)**n)*matern_slope_at_pi(nu, rho)*n**(-2.0)).min()
    nu_grid = np.linspace(0.28, 0.98, 60)
    theory = {r: np.array([pred_min(nu, r) for nu in nu_grid]) for r in (0.5, 1.0)}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axx = axes[0]
    ev, od = ns % 2 == 0, ns % 2 == 1
    nn = np.geomspace(60, 3000, 400)
    pe = Co0*nn**(-(2*nu0+1)) + ap0*nn**(-2); po = Co0*nn**(-(2*nu0+1)) - ap0*nn**(-2)
    axx.loglog(ns[od], vals[od], 'o', ms=4.5, color=PALETTE[0], label="odd $n$ (measured)")
    axx.loglog(nn, po, '-', lw=1.2, color=PALETTE[0], alpha=0.7, label="odd $n$ (two-term law)")
    pos = ev & (vals > 0); neg = ev & (vals < 0)
    axx.loglog(ns[pos], vals[pos], 's', ms=4.5, color=PALETTE[3], label="even $n$, positive")
    axx.loglog(ns[neg], -vals[neg], 's', ms=4.5, mfc='white', mec=PALETTE[3], label="even $n$, $|b_n|$ (negative)")
    axx.loglog(nn[pe > 0], pe[pe > 0], '-', lw=1.2, color=PALETTE[3], alpha=0.7)
    axx.loglog(nn[pe < 0], -pe[pe < 0], '--', lw=1.2, color=PALETTE[3], alpha=0.7, label="even $n$ (two-term law)")
    axx.axvline(nmin, color='black', lw=0.8, ls=':', label=r"predicted $n_{\min}=553$")
    axx.set_xlabel("degree $n$"); axx.set_ylabel("$|b_{n,2}|$")
    axx.set_title(r"Anatomy of the two-term law ($\nu=0.85$, $\rho=1$)", loc="left", fontweight="bold")
    axx.legend(frameon=False, loc="lower left", fontsize=7)
    axx = axes[1]
    axx.axvline(0.5, color='black', lw=1.0, ls=':', label=r"classical boundary $\nu=1/2$")
    axx.axvspan(0.5, 1.55, color='#c23b22', alpha=0.07)
    axx.plot(nus_s, m05, 'o', ms=5, color=PALETTE[0], label=r"scan minimum, $\rho=0.5$")
    axx.plot(nus_s, m10, 's', ms=5, color=PALETTE[3], label=r"scan minimum, $\rho=1.0$")
    axx.plot(nu_grid, theory[0.5], '-', lw=1.3, color=PALETTE[0], alpha=0.75, label=r"two-term prediction, $\rho=0.5$")
    axx.plot(nu_grid, theory[1.0], '-', lw=1.3, color=PALETTE[3], alpha=0.75, label=r"two-term prediction, $\rho=1.0$")
    axx.axhline(0, color='gray', lw=0.8)
    axx.set_yscale('symlog', linthresh=1e-9); axx.set_xlim(0.25, 1.55)
    axx.set_xlabel(r"Mat\'ern smoothness $\nu$"); axx.set_ylabel(r"$\min_{n\leq3000}\; b_{n,2}$")
    axx.set_title("Scan minima against the predicted law", loc="left", fontweight="bold")
    axx.legend(frameon=False, loc="lower left", fontsize=7)
    for k, a in enumerate(axes): a.text(-0.08, 1.02, "(%s)" % "ab"[k], transform=a.transAxes, fontweight="bold")
    fig.tight_layout(); fig.savefig('matern_calibration.pdf', bbox_inches='tight'); plt.close(fig)

if __name__ == "__main__":
    for fig, pdf, needs in [(fig_theorem, "theorem.pdf", None), (fig_planet, "planet.pdf", None),
                            (fig_experiments, "experiments.pdf", "exp_invisibility.py and exp_phase.py"),
                            (fig_matern, "matern_calibration.pdf", "exp_matern.py")]:
        try:
            fig(); print(pdf)
        except FileNotFoundError:
            print(f"{pdf} skipped (run {needs} first)")
