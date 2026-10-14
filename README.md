# Double_dot_potential_3D_rendering
Follow this repo to visualise double dot potential in 3D. Used claude 


## Installing in a virtual environment (Windows, Git Bash)

Create a virtual environment named `venv`, activate it, and install the requirements into it:

```bash
python -m venv venv
source venv/Scripts/activate
pip install -r requirements.txt
```

Your prompt shows `(venv)` while it is active. Run `deactivate` to leave it.
On macOS/Linux, activate with `source venv/bin/activate` instead.

## Usage

With the virtual environment active:

```bash
python double_dot_potential.py --model quartic --d 50 --eps 0.5
```

Writes `double_dot_surface.png` (3D surface at z = 0 plus a line cut along x) and
`double_dot_isosurface.html` (interactive 3D equipotential surfaces, needs plotly).

### Interactive mode (sliders)

```bash
python double_dot_potential.py --interactive
```

Opens a window with the 3D surface, a line cut along x, and knobs for:

- **cut y** – where the line cut is taken (shown as a red line on the surface)
- **slice z** – height of the plotted xy-plane
- **energy E** – energy level; drawn as a red contour on the surface and a dashed
  line on the cut, with the classically allowed region (V < E) shaded
- **detuning eps**, **half spacing d**, **hbar w**, **depth V0**, **width sigma**, **hbar wz**
- a **quartic / gaussian** model switch

Drag the 3D plot to rotate it; the view is kept while you move sliders.

Options: `--model quartic|gaussian`, `--d` half dot spacing (nm), `--hw` / `--hwz`
confinement energies (meV), `--eps` detuning (meV), `--m-eff` effective mass
(GaAs 0.067, Si 0.19). Run with `--help` for the full list.
