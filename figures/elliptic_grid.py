"""Figure 3.1: the tensor quadrature grid of the Euclidean basket in elliptic and
whitened coordinates. Writes p1_elliptic_grid.pdf to the working directory."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 9,
    "axes.linewidth": 0.7,
    "axes.titlesize": 9.5,
    "figure.dpi": 300,
})

s1, s2, rho, R = 0.20, 0.30, 0.50, 0.5
Sig = np.array([[s1 ** 2, rho * s1 * s2], [rho * s1 * s2, s2 ** 2]])
d = np.linalg.eigvalsh(Sig)                     # ascending
a, b = R / np.sqrt(d[0]), R / np.sqrt(d[1])     # semi-major, semi-minor
c = np.sqrt(a ** 2 - b ** 2)
mu0 = np.arctanh(b / a)

n_mu, n_nu = 10, 32
xg, _w = np.polynomial.legendre.leggauss(n_mu)
mu_nodes = 0.5 * mu0 * (xg + 1.0)
nu_nodes = np.arange(n_nu) * 2 * np.pi / n_nu
MU, NU = np.meshgrid(mu_nodes, nu_nodes, indexing="ij")
Z1 = c * np.cosh(MU) * np.cos(NU)
Z2 = c * np.sinh(MU) * np.sin(NU)
J = c ** 2 * (np.sinh(MU) ** 2 + np.sin(NU) ** 2)

cmap = plt.get_cmap("viridis")
norm = Normalize(vmin=0.0, vmax=J.max())
th = np.linspace(0, 2 * np.pi, 400)

fig, (ax_z, ax_e) = plt.subplots(1, 2, figsize=(7.2, 2.75),
                                 gridspec_kw=dict(width_ratios=[1.25, 1.0]))

ax_z.plot(a * np.cos(th), b * np.sin(th), color="#1d4ed8", lw=1.3, zorder=2)
ax_z.plot([-c, c], [0, 0], color="#6b7280", lw=0.8, zorder=1)
ax_z.plot([-c, c], [0, 0], "x", ms=4.5, mew=1.0, color="#111827", zorder=4)
sc = ax_z.scatter(Z1, Z2, c=J, cmap=cmap, norm=norm, s=7, lw=0, zorder=3)
ax_z.set_aspect("equal")
ax_z.set_xlim(-3.4, 3.4)
ax_z.set_ylim(-1.85, 1.85)
ax_z.set_xlabel(r"$z_1$", labelpad=1)
ax_z.set_ylabel(r"$z_2$", labelpad=1)
ax_z.tick_params(labelsize=7.5, length=2.5)
ax_z.set_title(r"(a) Nodes in the whitened plane")

ax_e.scatter(NU, MU, c=J, cmap=cmap, norm=norm, s=7, lw=0, zorder=3)
ax_e.axhline(mu0, color="#1d4ed8", lw=1.3, zorder=2)
ax_e.axhline(0.0, color="#6b7280", lw=0.8, zorder=2)
ax_e.text(2 * np.pi - 0.05, mu0 + 0.02, r"barrier $\mu=\mu_0$", ha="right", va="bottom",
          fontsize=7.5, color="#1d4ed8")
ax_e.text(2 * np.pi - 0.05, -0.012, r"focal segment $\mu=0$", ha="right", va="top",
          fontsize=7.5, color="#374151")
ax_e.set_xlim(-0.1, 2 * np.pi + 0.1)
ax_e.set_ylim(-0.085, mu0 + 0.11)
ax_e.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
ax_e.set_xticklabels([r"$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
ax_e.set_xlabel(r"$\nu$", labelpad=1)
ax_e.set_ylabel(r"$\mu$", labelpad=1)
ax_e.tick_params(labelsize=7.5, length=2.5)
ax_e.set_title(r"(b) The same nodes in $(\nu,\mu)$")

cb = fig.colorbar(sc, ax=[ax_z, ax_e], fraction=0.035, pad=0.02)
cb.set_label(r"area element $J=c^2(\sinh^2\mu+\sin^2\nu)$", fontsize=8)
cb.ax.tick_params(labelsize=7)

fig.savefig("p1_elliptic_grid.pdf", bbox_inches="tight")
fig.savefig("p1_elliptic_grid.png", bbox_inches="tight", dpi=200)
print(f"wrote p1_elliptic_grid.pdf/.png  (a={a:.3f}, b={b:.3f}, c={c:.3f}, mu0={mu0:.4f}, "
      f"J range {J.min():.3g}..{J.max():.3g})")
