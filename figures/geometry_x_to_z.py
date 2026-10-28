"""Figure 2.1: the three survival regions in log-return coordinates and after
whitening. Writes p1_geometry_x_to_z.pdf to the working directory."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyArrowPatch

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 9,
    "axes.linewidth": 0.7,
    "axes.titlesize": 9.5,
    "figure.dpi": 300,
})

WORK_FILL = "#dbe6fb"
WORK_EDGE = "#1d4ed8"
IDLE_FILL = "#eeeeee"
IDLE_EDGE = "#6b7280"
GRID = "#7c8aa5"
SPOT = "#b91c1c"
AXES_C = "#9ca3af"


def whitening(s1, s2, rho):
    """Return (Sigma, d, P) with d ascending, so that the image of a Euclidean disc
    has its major axis along z1 (horizontal in the plots)."""
    Sig = np.array([[s1 ** 2, rho * s1 * s2], [rho * s1 * s2, s2 ** 2]])
    d, P = np.linalg.eigh(Sig)          # ascending
    return Sig, d, P


def to_z(x, d, P):
    return (np.diag(d ** -0.5) @ P.T @ x)


def style(ax, xlabel, ylabel, lim):
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel(xlabel, labelpad=1)
    ax.set_ylabel(ylabel, labelpad=1)
    ax.tick_params(labelsize=7.5, length=2.5)
    ax.axhline(0, color=AXES_C, lw=0.4, zorder=0)
    ax.axvline(0, color=AXES_C, lw=0.4, zorder=0)
    ticks = [-0.5, 0.0, 0.5] if lim < 1 else [-3, -2, -1, 0, 1, 2, 3]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)


def spot(ax):
    ax.plot(0, 0, "o", ms=3.2, color=SPOT, zorder=6)


def principal_axes(ax, d, P, length):
    for k in range(2):
        v = P[:, k] * length
        ax.plot([-v[0], v[0]], [-v[1], v[1]], ls=(0, (3, 2)), lw=0.6,
                color="#374151", zorder=3)


fig = plt.figure(figsize=(7.2, 5.75))
gs = fig.add_gridspec(2, 3, left=0.105, right=0.985, top=0.925, bottom=0.07,
                      hspace=0.62, wspace=0.42)
axes = np.array([[fig.add_subplot(gs[r, c]) for c in range(3)] for r in range(2)])
th = np.linspace(0, 2 * np.pi, 400)

# ── Column 1: double no-touch rectangle ─────────────────────────────
s1, s2, rho = 0.20, 0.25, 0.40
_, d, P = whitening(s1, s2, rho)
x1lo, x1hi = np.log(80 / 100), np.log(125 / 100)
x2lo, x2hi = np.log(75 / 100), np.log(130 / 100)
corners = np.array([[x1lo, x2lo], [x1hi, x2lo], [x1hi, x2hi], [x1lo, x2hi]]).T

ax = axes[0, 0]
ax.add_patch(Polygon(corners.T, closed=True, fc=WORK_FILL, ec=WORK_EDGE, lw=1.3, zorder=2))
for v in np.linspace(x1lo, x1hi, 7)[1:-1]:
    ax.plot([v, v], [x2lo, x2hi], color=GRID, lw=0.4, alpha=0.7, zorder=2)
for v in np.linspace(x2lo, x2hi, 7)[1:-1]:
    ax.plot([x1lo, x1hi], [v, v], color=GRID, lw=0.4, alpha=0.7, zorder=2)
spot(ax)
style(ax, r"$x_1$", r"$x_2$", 0.62)
ax.set_title("(a) Double no-touch:\nrectangle, Fourier–Galerkin in $x$")

ax = axes[1, 0]
zc = to_z(corners, d, P)
ax.add_patch(Polygon(zc.T, closed=True, fc=IDLE_FILL, ec=IDLE_EDGE, lw=1.1,
                     ls=(0, (4, 2)), zorder=2))
for v in np.linspace(x1lo, x1hi, 7)[1:-1]:
    seg = to_z(np.array([[v, v], [x2lo, x2hi]]), d, P)
    ax.plot(seg[0], seg[1], color=GRID, lw=0.4, alpha=0.7, zorder=2)
for v in np.linspace(x2lo, x2hi, 7)[1:-1]:
    seg = to_z(np.array([[x1lo, x1hi], [v, v]]), d, P)
    ax.plot(seg[0], seg[1], color=GRID, lw=0.4, alpha=0.7, zorder=2)
spot(ax)
style(ax, r"$z_1$", r"$z_2$", 3.4)
ax.set_title("(d) Image in $z$: parallelogram\n(not used)")

# ── Column 2: Mahalanobis ellipse ─────────────────────────────────
s1, s2, rho = 0.20, 0.30, 0.50
Sig, d, P = whitening(s1, s2, rho)
R_M = 1.5
circ = np.vstack([np.cos(th), np.sin(th)])
x_ell = P @ np.diag(d ** 0.5) @ (R_M * circ)          # boundary in x

ax = axes[0, 1]
ax.add_patch(Polygon(x_ell.T, closed=True, fc=WORK_FILL, ec=WORK_EDGE, lw=1.3, zorder=2))
principal_axes(ax, d, P, 0.6)
spot(ax)
style(ax, r"$x_1$", r"$x_2$", 0.62)
ax.set_title("(b) Mahalanobis barrier:\n" r"$\mathbf{x}^\top\Sigma^{-1}\mathbf{x}\leq R^2$")

ax = axes[1, 1]
ax.add_patch(Polygon((R_M * circ).T, closed=True, fc=WORK_FILL, ec=WORK_EDGE, lw=1.3, zorder=2))
for rr in np.linspace(0, R_M, 5)[1:-1]:
    ax.plot(rr * np.cos(th), rr * np.sin(th), color=GRID, lw=0.4, alpha=0.7, zorder=2)
for a in np.linspace(0, 2 * np.pi, 13)[:-1]:
    ax.plot([0, R_M * np.cos(a)], [0, R_M * np.sin(a)], color=GRID, lw=0.4, alpha=0.7, zorder=2)
spot(ax)
style(ax, r"$z_1$", r"$z_2$", 3.4)
ax.set_title("(e) Image in $z$: disc,\npolar coordinates, Bessel")

# ── Column 3: Euclidean disc ─────────────────────────────────────
R_E = 0.5
ax = axes[0, 2]
ax.add_patch(Polygon((R_E * circ).T, closed=True, fc=WORK_FILL, ec=WORK_EDGE, lw=1.3, zorder=2))
principal_axes(ax, d, P, 0.6)
spot(ax)
style(ax, r"$x_1$", r"$x_2$", 0.62)
ax.set_title("(c) Euclidean barrier:\n" r"$x_1^2+x_2^2\leq R^2$")

ax = axes[1, 2]
a_semi, b_semi = R_E / np.sqrt(d[0]), R_E / np.sqrt(d[1])   # major along z1
c = np.sqrt(a_semi ** 2 - b_semi ** 2)
mu0 = np.arctanh(b_semi / a_semi)
ax.add_patch(Polygon(np.vstack([a_semi * np.cos(th), b_semi * np.sin(th)]).T, closed=True,
                     fc=WORK_FILL, ec=WORK_EDGE, lw=1.3, zorder=2))
for mu in np.linspace(0, mu0, 5)[1:-1]:
    ax.plot(c * np.cosh(mu) * np.cos(th), c * np.sinh(mu) * np.sin(th),
            color=GRID, lw=0.4, alpha=0.7, zorder=2)
mus = np.linspace(0, mu0, 60)
for nu in np.linspace(0, 2 * np.pi, 17)[:-1]:
    ax.plot(c * np.cosh(mus) * np.cos(nu), c * np.sinh(mus) * np.sin(nu),
            color=GRID, lw=0.4, alpha=0.7, zorder=2)
ax.plot([-c, c], [0, 0], color=GRID, lw=0.6, zorder=2)
ax.plot([-c, c], [0, 0], "x", ms=4, mew=1.0, color="#111827", zorder=5)
spot(ax)
style(ax, r"$z_1$", r"$z_2$", 3.4)
ax.set_title("(f) Image in $z$: ellipse,\nelliptic coordinates, Mathieu")

# Arrows between the rows, and a label for the map
for col in range(3):
    top = axes[0, col].get_position()
    bot = axes[1, col].get_position()
    xm = 0.5 * (top.x0 + top.x1)
    y_hi = top.y0 - 0.062
    y_lo = bot.y1 + 0.078
    fig.add_artist(FancyArrowPatch((xm, y_hi), (xm, y_lo),
                                   transform=fig.transFigure, arrowstyle="-|>",
                                   mutation_scale=10, lw=1.0, color="#374151"))
gap_mid = 0.5 * (axes[0, 0].get_position().y0 + axes[1, 0].get_position().y1) + 0.006
fig.text(0.5 * (axes[0, 0].get_position().x1 + axes[0, 1].get_position().x0), gap_mid,
         "whitening", ha="center", va="center", fontsize=8.5, style="italic", color="#374151")
fig.text(0.5 * (axes[0, 1].get_position().x1 + axes[0, 2].get_position().x0), gap_mid,
         r"$\mathbf{z}=D^{-1/2}P^\top\mathbf{x}$", ha="center", va="center",
         fontsize=8.5, color="#374151")
fig.text(0.012, 0.5 * (axes[0, 0].get_position().y0 + axes[0, 0].get_position().y1),
         "log-return coordinates", rotation=90, va="center", ha="left",
         fontsize=8.5, color="#374151")
fig.text(0.012, 0.5 * (axes[1, 0].get_position().y0 + axes[1, 0].get_position().y1),
         "whitened coordinates", rotation=90, va="center", ha="left",
         fontsize=8.5, color="#374151")

fig.savefig("p1_geometry_x_to_z.pdf", bbox_inches="tight")
fig.savefig("p1_geometry_x_to_z.png", bbox_inches="tight", dpi=200)
print("wrote p1_geometry_x_to_z.pdf/.png")
