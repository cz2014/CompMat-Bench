#!/bin/bash
# Extract the isotropic exchange between the Cr sites from the Wannier
# Hamiltonian of the `wannier` step. Run in that step's directory.
set -euo pipefail

wann2J.py \
  --path . \
  --posfile POSCAR \
  --prefix_up wannier90.1 \
  --prefix_down wannier90.2 \
  --elements Cr \
  --efermi 9.6310 \
  --kmesh 6 12 8 \
  --emin -14.0 \
  --emax 0.0 \
  --nz 100 \
  --rcut 6.0 \
  --np 32 \
  --output_path TB2J_results \
  --description 'isotropic exchange from the Wannier Hamiltonian'
