# Task: effective thermal conductivity of finite Si and Ge slabs from phonon Monte Carlo run outputs

Phonon heat conduction in crystalline Si and Ge has been simulated with a
mode-resolved, deviational Monte Carlo solution of the phonon Boltzmann
transport equation in the relaxation-time approximation. The geometry is a
homogeneous slab of length $L$ along $x$ between two isothermal reservoirs at
$x = 0$ and $x = L$: a phonon that reaches a reservoir is absorbed, and the
reservoir emits phonons from its own equilibrium distribution. Nothing else
scatters phonons besides the intrinsic phonon-phonon processes of the bulk
crystal. Each run records its steady state: the heat flux density through the
slab and the temperature profile across it. The question is how the apparent
conductivity of such a slab depends on its length, and how much of the
temperature difference imposed by the reservoirs drops inside the
slab. Two lengths of a longer sweep are given here.

## What you are given

- `runs/<material>/<L>nm/run_<k>.json` -- five independent runs (`k` = 1 to
  5, different random seeds) for each material (`Si`, `Ge`) and each slab
  length (100 nm and 2000 nm). Each file holds the material, the slab length,
  the temperatures of the two reservoirs, the steady-state heat flux density
  through the slab, and the steady-state temperature averaged over each of
  the equal-width cells that tile the whole slab from the hot reservoir face
  to the cold one, in that order. Units are part of the key names.

No simulation is required. The environment provides Python with numpy.

## Definitions (fixed for this task)

Report the reservoir-to-reservoir effective conductivity and the interior
drop fraction. Both quantities use the imposed reservoir temperature
difference as their temperature reference.

The interior drop is defined by the least-squares temperature gradient over
the central half of the slab ($L/4 \le x \le 3L/4$), continued across the
full slab length from the hot face to the cold face. Cell temperatures are
located at cell centres. The interior drop fraction is this drop relative
to the imposed reservoir temperature difference.

Each reported value is the mean of the values determined separately for
the five runs of that material and length.

## What to determine

(1) $\kappa_\mathrm{eff}$ for Si and Ge at 100 nm and at 2000 nm.

(2) The interior drop fraction for the same four cases.

## Output format

Write a single `answer.json` in the workspace root with exactly these fields:

```json
{
  "kappa_eff_Si_100nm_W_per_mK": <float>,
  "kappa_eff_Si_2000nm_W_per_mK": <float>,
  "kappa_eff_Ge_100nm_W_per_mK": <float>,
  "kappa_eff_Ge_2000nm_W_per_mK": <float>,
  "interior_drop_fraction_Si_100nm": <float>,
  "interior_drop_fraction_Si_2000nm": <float>,
  "interior_drop_fraction_Ge_100nm": <float>,
  "interior_drop_fraction_Ge_2000nm": <float>
}
```

- `kappa_eff_*`: item (1), in $\mathrm{W\,m^{-1}\,K^{-1}}$.
- `interior_drop_fraction_*`: item (2), dimensionless.

All values must be computed from the staged files; do not guess or hardcode.
