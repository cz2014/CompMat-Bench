#!/usr/bin/env bash
# Self-consistent Hartree solve at n_tot = +2.0e11 cm^-2 = +0.002 e/nm^2.
# Re-expressed from 3_reproduction/L2/drivers/run_chain2.sh with literal
# values; selfcon budget 60 covers a cold solve (the as-run production chain
# warm-started this solve from the converged symmetric potential, with the
# warm-start file argument placed last on the command line).
set -u
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg

kdotpy 2d 8o k 0 1.0 / 0.02 kphi 0 90 / 5 \
  msubst CdTe mlayer HgCdTe 70% HgTe HgCdTe 70% llayer 55 45 55 zres 0.25 \
  targetenergy 0 neig 120 erange -60 150 split 0.01 noax \
  matparam "HgTe:renormalize_EP=20800; CdTe:renormalize_EP=20800; CdTe:Ev=-620; HgCdTe:Ev=-434.0; HgCdTe:diel_epsilon=13.6" \
  efield 1.198 - cardens 0.002 selfcon 60 0.01 \
  config "selfcon_full_diag=false;selfcon_acceptable_status=2" \
  obs z dos cpus 20 out .p2 outdir .
