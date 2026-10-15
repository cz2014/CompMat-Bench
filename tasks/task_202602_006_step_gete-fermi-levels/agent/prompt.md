# Dirac-point reference and Fermi levels of hole-doped rhombohedral GeTe under an in-plane exchange field

Ferroelectric rhombohedral GeTe is a giant-Rashba semiconductor whose valence
bands host a Kramers-degenerate Dirac crossing at the zone-boundary point Z.
You are given a spinor Wannier tight-binding model and its spin operator. Real samples are p-type; your task is to
place the Fermi level of three experimentally measured hole densities on this
model's energy scale, referenced to the Dirac point, in the presence of a
fixed in-plane exchange field.

## The provided model

`bundle/` contains the Wannier model data. `wannier_bundle.py` provides the
model interface, documented in its module docstring.

## The exchange field

The field-on model (used in (b) below; (a) is zero-field) carries a Zeeman
term representing the samples' exchange field: -50 meV along the [11-2]
crystal direction -- components (-50, 0, 0) meV in the orthonormal
([11-2], [1-10], [111]) crystal frame, using the supplied module's
Zeeman convention.

## What to compute

(a) **The Dirac-point energy `E_DP`** of the **zero-field** model: the energy
at which the Rashba-split topmost valence bands cross at Z = (0.5, 0.5, 0.5)
(fractional coordinates) -- a Kramers degeneracy of the zero-field model.

(b) **Three Fermi energies of the field-on model**, one for each measured
hole density

    p = 1.1e21, 5.3e20, 1.8e19 cm^-3,

at zero temperature with no smearing, with hole densities referenced to the
charge-neutral filling of the supplied model. Use the uniform 80 x 80 x 80
fractional k-mesh that includes k = 0 (i.e. k_i in {0, 1/80, ..., 79/80});
report the values from exactly this mesh.

(c) **Report `E_F - E_DP` in eV** for each density.

## Answer format

Write a single `answer.json` in the workspace root with exactly:

```json
{
  "E_DP_eV": <float>,
  "EF_minus_EDP_p1.1e21_eV": <float>,
  "EF_minus_EDP_p5.3e20_eV": <float>,
  "EF_minus_EDP_p1.8e19_eV": <float>,
  "cell_volume_A3": <float>
}
```

`E_DP_eV` is the absolute model energy from (a); the three `EF_minus_EDP_*`
fields are the signed differences from (c) in eV; `cell_volume_A3` is the
unit-cell volume (Angstrom^3) you used for the density-to-holes conversion.
All fields are required. Numbers only; no prose is read.
