"""
Electrostatic potential beneath a grid of rectangular surface gates.

Based on Eqs. (10)-(11) of Anderson et al., "High-precision real-space simulation
of electrostatically-confined few-electron states", arXiv:2203.00082, which give
the potential under a row of square gates (side 2a, centres x_i):

    phi0(x,y,z) = -sum_i V_i [ g(a - x_i + x, a + y, z) + g(a - x_i + x, a - y, z)
                             + g(a + x_i - x, a + y, z) + g(a + x_i - x, a - y, z) ]

    g(u,v,z) = (1 / 2pi) * atan( u v / (z sqrt(u^2 + v^2 + z^2)) )

Here each gate i is generalised to a rectangle centred at (x_i, y_i) with
half-widths (ax_i, ay_i). With X = x - x_i and Y = y - y_i:

    phi0 = -sum_i V_i [ g(ax_i + X, ay_i + Y, z) + g(ax_i + X, ay_i - Y, z)
                      + g(ax_i - X, ay_i + Y, z) + g(ax_i - X, ay_i - Y, z) ]

The surface outside the gates is held at 0 V. As in the paper, phi0 is the
electron potential energy (eV per volt of gate bias), so positive gates make wells.
It is plotted here in meV. z is the depth below the gates (z > 0).

Usage:
  python gate_grid_potential.py                       # paper's 3-gate double dot
  python gate_grid_potential.py --gates gates.json    # load your own gate grid

Gate files are JSON lists of {"x": nm, "y": nm, "ax": nm, "ay": nm, "V": volts}.
In the window, pick a gate with the arrows or by clicking it in the top view,
then move its sliders. "3D volume" opens a see-through volume rendering of
phi0(x, y, z) in the browser (needs plotly).
"""

import argparse
import json

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button, Slider

# Paper's Fig. 1 example: three 70 nm square gates, P1 / X / P2
DEFAULT_GATES = [
    {"x": -70.0, "y": 0.0, "ax": 35.0, "ay": 35.0, "V": 0.25},
    {"x": 0.0, "y": 0.0, "ax": 35.0, "ay": 35.0, "V": 0.10},
    {"x": 70.0, "y": 0.0, "ax": 35.0, "ay": 35.0, "V": 0.25},
]
DEFAULT_DEPTH = 70.0  # nm, z-bar in the paper

# Volume rendering: wells (low energy) opaque, high-energy background nearly clear,
# so the dots inside the volume stay visible.
VOLUME_COLORSCALE = "Turbo"
VOLUME_OPACITYSCALE = [[0.0, 0.8], [0.4, 0.35], [0.75, 0.12], [1.0, 0.03]]
SURFACE_CMAP = cm.turbo


def g(u, v, z):
    """Eq. (11)."""
    return np.arctan(u * v / (z * np.sqrt(u**2 + v**2 + z**2))) / (2 * np.pi)


def gate_potential(x, y, z, gates):
    """Eq. (10) generalised to rectangles. Returns potential energy in meV."""
    phi = np.zeros(np.broadcast(x, y, z).shape)
    for gt in gates:
        X, Y = x - gt["x"], y - gt["y"]
        ax, ay = gt["ax"], gt["ay"]
        phi -= gt["V"] * (g(ax + X, ay + Y, z) + g(ax + X, ay - Y, z)
                          + g(ax - X, ay + Y, z) + g(ax - X, ay - Y, z))
    return 1e3 * phi


def plot_extent(gates, depth):
    """Square-ish window around all gates with a margin of one depth."""
    xs = [gt["x"] - gt["ax"] for gt in gates] + [gt["x"] + gt["ax"] for gt in gates]
    ys = [gt["y"] - gt["ay"] for gt in gates] + [gt["y"] + gt["ay"] for gt in gates]
    m = max(depth, 30.0)
    return min(xs) - m, max(xs) + m, min(ys) - m, max(ys) + m


def render_volume(gates, depth, n=48, filename="gate_grid_volume.html"):
    """See-through 3D volume rendering of phi0(x, y, z) with plotly."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("plotly not installed; pip install plotly")
        return

    x0, x1, y0, y1 = plot_extent(gates, depth)
    x = np.linspace(x0, x1, n)
    y = np.linspace(y0, y1, n)
    z = np.linspace(0.4 * depth, 1.6 * depth, n // 2)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    V = gate_potential(X, Y, Z, gates)

    fig = go.Figure(go.Volume(
        x=X.ravel(), y=Y.ravel(), z=-Z.ravel(), value=V.ravel(),
        # Drop the flat high-energy background so it doesn't hide the wells
        isomin=V.min(), isomax=np.percentile(V, 85), surface_count=20,
        colorscale=VOLUME_COLORSCALE, opacityscale=VOLUME_OPACITYSCALE,
        caps=dict(x_show=False, y_show=False, z_show=False),
        colorbar=dict(title="U (meV)"),
    ))
    # Gate outlines on the surface (z = 0)
    for i, gt in enumerate(gates):
        cx = [gt["x"] - gt["ax"], gt["x"] + gt["ax"], gt["x"] + gt["ax"], gt["x"] - gt["ax"], gt["x"] - gt["ax"]]
        cy = [gt["y"] - gt["ay"], gt["y"] - gt["ay"], gt["y"] + gt["ay"], gt["y"] + gt["ay"], gt["y"] - gt["ay"]]
        fig.add_trace(go.Scatter3d(x=cx, y=cy, z=[0] * 5, mode="lines",
                                   line=dict(color="black", width=4),
                                   name=f"gate {i} ({gt['V']:.2f} V)"))
    fig.update_layout(
        title="Potential energy beneath the gate grid (gates at z = 0)",
        legend=dict(x=0.0, y=1.0),
        scene=dict(xaxis_title="x (nm)", yaxis_title="y (nm)",
                   zaxis_title="-depth (nm)", aspectmode="data"),
    )
    fig.write_html(filename)
    print(f"Saved {filename}")
    fig.show()


class GateGridApp:
    def __init__(self, gates, depth, gates_out):
        self.gates = [dict(gt) for gt in gates]
        self.sel = 0
        self.gates_out = gates_out
        self._syncing = False

        self.fig = plt.figure(figsize=(16, 9))
        self.fig.subplots_adjust(left=0.05, right=0.98, top=0.95, bottom=0.36, wspace=0.25)
        self.ax_top = self.fig.add_subplot(1, 3, 1)
        self.ax_3d = self.fig.add_subplot(1, 3, 2, projection="3d")
        self.ax_cut = self.fig.add_subplot(1, 3, 3)
        self.ax_3d.view_init(elev=35, azim=-60)
        self.cbar = None

        def slider(row, label, lo, hi, init, left=0.12, width=0.38):
            sax = self.fig.add_axes([left, 0.27 - row * 0.04, width, 0.025])
            return Slider(sax, label, lo, hi, valinit=init)

        g0 = self.gates[0]
        self.s_x = slider(0, "x_i (nm)", -400, 400, g0["x"])
        self.s_y = slider(1, "y_i (nm)", -400, 400, g0["y"])
        self.s_ax = slider(2, "ax_i half-width (nm)", 5, 200, g0["ax"])
        self.s_ay = slider(3, "ay_i half-width (nm)", 5, 200, g0["ay"])
        self.s_v = slider(4, "V_i (V)", -0.5, 1.0, g0["V"])
        self.s_z = slider(0, "depth z (nm)", 5, 200, depth, left=0.62, width=0.3)
        self.s_cut = slider(1, "cut y (nm)", -400, 400, 0.0, left=0.62, width=0.3)
        for s in (self.s_x, self.s_y, self.s_ax, self.s_ay, self.s_v):
            s.on_changed(self.on_gate_slider)
        for s in (self.s_z, self.s_cut):
            s.on_changed(lambda _: self.redraw())

        def button(i, label, cb):
            b = Button(self.fig.add_axes([0.62 + i * 0.062, 0.06, 0.058, 0.045]), label)
            b.on_clicked(cb)
            return b

        self.buttons = [
            button(0, "< gate", lambda _: self.select(self.sel - 1)),
            button(1, "gate >", lambda _: self.select(self.sel + 1)),
            button(2, "add gate", self.add_gate),
            button(3, "remove", self.remove_gate),
            button(4, "save", self.save),
            button(5, "3D volume", lambda _: render_volume(self.gates, self.s_z.val)),
        ]
        self.status = self.fig.text(0.62, 0.13, "", fontsize=11)
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.redraw()

    # --- gate editing -------------------------------------------------------
    def on_gate_slider(self, _):
        if self._syncing:
            return
        gt = self.gates[self.sel]
        gt.update(x=self.s_x.val, y=self.s_y.val, ax=self.s_ax.val,
                  ay=self.s_ay.val, V=self.s_v.val)
        self.redraw()

    def select(self, i):
        self.sel = i % len(self.gates)
        gt = self.gates[self.sel]
        self._syncing = True  # don't write back while moving sliders to this gate
        for s, k in ((self.s_x, "x"), (self.s_y, "y"), (self.s_ax, "ax"),
                     (self.s_ay, "ay"), (self.s_v, "V")):
            s.set_val(gt[k])
        self._syncing = False
        self.redraw()

    def add_gate(self, _):
        last = self.gates[self.sel]
        self.gates.append({"x": last["x"] + 2 * last["ax"], "y": last["y"],
                           "ax": last["ax"], "ay": last["ay"], "V": last["V"]})
        self.select(len(self.gates) - 1)

    def remove_gate(self, _):
        if len(self.gates) > 1:
            self.gates.pop(self.sel)
            self.select(min(self.sel, len(self.gates) - 1))

    def save(self, _):
        with open(self.gates_out, "w") as f:
            json.dump(self.gates, f, indent=2)
        self.status.set_text(f"Saved {len(self.gates)} gates to {self.gates_out}")
        self.fig.canvas.draw_idle()

    def on_click(self, event):
        if event.inaxes is not self.ax_top or event.xdata is None:
            return
        for i, gt in enumerate(self.gates):
            if abs(event.xdata - gt["x"]) <= gt["ax"] and abs(event.ydata - gt["y"]) <= gt["ay"]:
                self.select(i)
                return

    # --- drawing ------------------------------------------------------------
    def redraw(self):
        depth, ycut = self.s_z.val, self.s_cut.val
        x0, x1, y0, y1 = plot_extent(self.gates, depth)
        x = np.linspace(x0, x1, 140)
        y = np.linspace(y0, y1, 140)
        X, Y = np.meshgrid(x, y)
        U = gate_potential(X, Y, depth, self.gates)
        cut = gate_potential(x, ycut, depth, self.gates)

        # Top view with gate outlines
        ax = self.ax_top
        ax.cla()
        im = ax.imshow(U, extent=(x0, x1, y0, y1), origin="lower", cmap=SURFACE_CMAP, aspect="equal")
        if self.cbar is None:
            self.cbar = self.fig.colorbar(im, ax=ax, fraction=0.046, label="U (meV)")
        else:
            self.cbar.update_normal(im)
        for i, gt in enumerate(self.gates):
            chosen = i == self.sel
            ax.add_patch(Rectangle((gt["x"] - gt["ax"], gt["y"] - gt["ay"]), 2 * gt["ax"], 2 * gt["ay"],
                                   fill=False, edgecolor="white" if chosen else "black",
                                   linewidth=2.5 if chosen else 1.0, linestyle="-" if chosen else "--"))
            ax.text(gt["x"], gt["y"], f"{i}\n{gt['V']:.2f} V", ha="center", va="center",
                    color="white" if chosen else "black", fontsize=8)
        ax.axhline(ycut, color="white", linewidth=1, linestyle=":")
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_xlabel("x (nm)")
        ax.set_ylabel("y (nm)")
        ax.set_title(f"Top view at depth {depth:.0f} nm (click a gate)")

        # 3D surface, semi-transparent so the far side stays visible
        elev, azim = self.ax_3d.elev, self.ax_3d.azim
        a3 = self.ax_3d
        a3.cla()
        a3.plot_surface(X, Y, U, cmap=SURFACE_CMAP, rstride=2, cstride=2,
                        linewidth=0.2, edgecolor=(0, 0, 0, 0.15), alpha=0.75)
        a3.contour(X, Y, U, levels=15, zdir="z", offset=U.min(), cmap=SURFACE_CMAP)
        a3.plot(x, np.full_like(x, ycut), cut, color="black", linewidth=2)
        a3.set_xlabel("x (nm)")
        a3.set_ylabel("y (nm)")
        a3.set_zlabel("U (meV)")
        a3.set_title("U(x, y) at depth z")
        a3.view_init(elev=elev, azim=azim)

        # Line cut
        c = self.ax_cut
        c.cla()
        c.plot(x, cut, color="C0")
        for i, gt in enumerate(self.gates):
            if abs(ycut - gt["y"]) <= gt["ay"]:
                c.axvspan(gt["x"] - gt["ax"], gt["x"] + gt["ax"],
                          color="C1" if i == self.sel else "0.85", alpha=0.3)
        c.set_xlim(x0, x1)
        c.set_xlabel("x (nm)")
        c.set_ylabel("U (meV)")
        c.set_title(f"Line cut at y = {ycut:.0f} nm (shaded: gates)")
        c.grid(alpha=0.3)

        self.status.set_text(f"Editing gate {self.sel} of {len(self.gates)}  "
                             f"(min U = {U.min():.1f} meV)")
        self.fig.canvas.draw_idle()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gates", help="JSON file with the gate list (default: paper's 3-gate example)")
    p.add_argument("--depth", type=float, default=DEFAULT_DEPTH, help="depth below gates (nm)")
    p.add_argument("--save-to", default="gates.json", help="where the 'save' button writes gates")
    p.add_argument("--volume", action="store_true", help="only render the 3D volume, no window")
    args = p.parse_args()

    gates = DEFAULT_GATES
    if args.gates:
        with open(args.gates) as f:
            gates = json.load(f)

    if args.volume:
        render_volume(gates, args.depth)
        return

    app = GateGridApp(gates, args.depth, args.save_to)  # noqa: F841 (keeps widgets alive)
    plt.show()


if __name__ == "__main__":
    main()
