"""
3D rendering of a double quantum dot confinement potential.

Two models are available:
  - "quartic":  V(x,y,z) = (m w^2 / 2) * [ (x^2 - d^2)^2 / (4 d^2) + y^2 ] + (m wz^2 / 2) z^2
                The standard biquadratic double-well (Burkard, Loss, DiVincenzo).
  - "gaussian": V(x,y,z) = -V0 * [ G(x+d) + G(x-d) ] + (m wz^2 / 2) z^2
                Two Gaussian wells, closer to gate-defined dots.

A detuning term eps * x / (2d) tilts the double well (left/right dot energy offset).

Outputs:
  1. 3D surface of the in-plane potential V(x, y, z=0)
  2. 3D isosurface plot of V(x, y, z) (plotly, if installed; otherwise skipped)

Usage:
  pip install numpy matplotlib plotly scikit-image
  python double_dot_potential.py --model quartic --d 50 --eps 0.5
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# Physical constants (SI)
HBAR = 1.054571817e-34
M_E = 9.1093837015e-31
MEV = 1.602176634e-22  # 1 meV in J
NM = 1e-9


def potential(x, y, z, model, d, hw, hwz, v0, sigma, eps, m_eff):
    """Return V in meV. x, y, z, d, sigma in nm; hw, hwz, v0, eps in meV."""
    m = m_eff * M_E
    # Convert confinement energies to angular frequencies
    w = hw * MEV / HBAR
    wz = hwz * MEV / HBAR
    xs, ys, zs, ds = x * NM, y * NM, z * NM, d * NM

    vz = 0.5 * m * wz**2 * zs**2 / MEV

    if model == "quartic":
        vxy = 0.5 * m * w**2 * ((xs**2 - ds**2) ** 2 / (4 * ds**2) + ys**2) / MEV
    elif model == "gaussian":
        g = lambda x0: np.exp(-((x - x0) ** 2 + y**2) / (2 * sigma**2))
        vxy = -v0 * (g(-d) + g(d))
        vxy -= vxy.min() if np.ndim(vxy) else 0.0
    else:
        raise ValueError(f"unknown model: {model}")

    detuning = eps * x / (2 * d)
    return vxy + vz + detuning


def plot_surface(args):
    x = np.linspace(-2.2 * args.d, 2.2 * args.d, 300)
    y = np.linspace(-1.5 * args.d, 1.5 * args.d, 200)
    X, Y = np.meshgrid(x, y)
    V = potential(X, Y, 0.0, args.model, args.d, args.hw, args.hwz,
                  args.v0, args.sigma, args.eps, args.m_eff)
    V = np.clip(V, None, args.vmax)

    fig = plt.figure(figsize=(12, 5))

    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.plot_surface(X, Y, V, cmap=cm.viridis, rstride=2, cstride=2,
                    linewidth=0, antialiased=True, alpha=0.95)
    ax.contour(X, Y, V, levels=20, zdir="z", offset=V.min(), cmap=cm.viridis)
    ax.set_xlabel("x (nm)")
    ax.set_ylabel("y (nm)")
    ax.set_zlabel("V (meV)")
    ax.set_title(f"Double dot potential, z = 0 ({args.model})")
    ax.view_init(elev=30, azim=-60)

    ax2 = fig.add_subplot(1, 2, 2)
    line = potential(x, 0.0, 0.0, args.model, args.d, args.hw, args.hwz,
                     args.v0, args.sigma, args.eps, args.m_eff)
    ax2.plot(x, line, color="C0")
    ax2.set_xlabel("x (nm)")
    ax2.set_ylabel("V (meV)")
    ax2.set_title("Cut along y = 0, z = 0")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig("double_dot_surface.png", dpi=150)
    print("Saved double_dot_surface.png")


def plot_isosurface(args):
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("plotly not installed; skipping 3D isosurface (pip install plotly)")
        return

    n = 60
    x = np.linspace(-2.2 * args.d, 2.2 * args.d, n)
    y = np.linspace(-1.5 * args.d, 1.5 * args.d, n)
    z = np.linspace(-0.5 * args.d, 0.5 * args.d, n // 2)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    V = potential(X, Y, Z, args.model, args.d, args.hw, args.hwz,
                  args.v0, args.sigma, args.eps, args.m_eff)

    vmin = V.min()
    fig = go.Figure(go.Isosurface(
        x=X.ravel(), y=Y.ravel(), z=Z.ravel(), value=V.ravel(),
        isomin=vmin + 0.5, isomax=vmin + args.vmax / 2,
        surface_count=5, opacity=0.35, colorscale="Viridis",
        caps=dict(x_show=False, y_show=False, z_show=False),
        colorbar=dict(title="V (meV)"),
    ))
    fig.update_layout(
        title=f"Double dot equipotential surfaces ({args.model})",
        scene=dict(xaxis_title="x (nm)", yaxis_title="y (nm)",
                   zaxis_title="z (nm)", aspectmode="data"),
    )
    fig.write_html("double_dot_isosurface.html")
    print("Saved double_dot_isosurface.html")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=["quartic", "gaussian"], default="quartic")
    p.add_argument("--d", type=float, default=50.0, help="half inter-dot distance (nm)")
    p.add_argument("--hw", type=float, default=3.0, help="in-plane confinement hbar*w (meV)")
    p.add_argument("--hwz", type=float, default=10.0, help="vertical confinement hbar*wz (meV)")
    p.add_argument("--v0", type=float, default=10.0, help="Gaussian well depth (meV)")
    p.add_argument("--sigma", type=float, default=25.0, help="Gaussian well width (nm)")
    p.add_argument("--eps", type=float, default=0.0, help="detuning (meV)")
    p.add_argument("--m-eff", type=float, default=0.067, help="effective mass (GaAs: 0.067, Si: 0.19)")
    p.add_argument("--vmax", type=float, default=15.0, help="clip potential above this (meV)")
    p.add_argument("--no-show", action="store_true", help="don't open matplotlib window")
    args = p.parse_args()

    plot_surface(args)
    plot_isosurface(args)
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
