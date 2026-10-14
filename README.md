# Double_dot_potential_3D_rendering
Follow this repo to visualise double dot potential in 3D. Used claude 


## Usage

```bash
pip install -r requirements.txt
python double_dot_potential.py --model quartic --d 50 --eps 0.5
```

Writes `double_dot_surface.png` (3D surface at z = 0 plus a line cut along x) and
`double_dot_isosurface.html` (interactive 3D equipotential surfaces, needs plotly).

Options: `--model quartic|gaussian`, `--d` half dot spacing (nm), `--hw` / `--hwz`
confinement energies (meV), `--eps` detuning (meV), `--m-eff` effective mass
(GaAs 0.067, Si 0.19). Run with `--help` for the full list.
