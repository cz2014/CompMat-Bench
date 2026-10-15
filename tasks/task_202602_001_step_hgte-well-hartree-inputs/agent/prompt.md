# Task: write the kdotpy run set for a self-consistent Hartree sweep of a gated HgTe quantum well

A thick HgTe quantum well under tensile strain realizes the semimetallic
three-dimensional topological-insulator phase. In a top-gated device the
carrier density is tuned across a range where even small electrostatic
potentials reshape the band structure, so the electrostatics must be solved
self-consistently: the multi-band k.p Hamiltonian and the Poisson equation
are iterated to a common solution at fixed total carrier density (a
Schrodinger-Poisson, or self-consistent Hartree, calculation).

Your job is to write the complete run specification -- the exact command
lines -- for three such self-consistent solves with the open-source band
structure package kdotpy (version 1.4.1; its source tree is provided, see
below). You do NOT run them; the commands are graded as written.

## Provided files and environment

- `references/kdotpy-1.4.1/`: the complete source tree of kdotpy version 1.4.1 (pure
  Python). Its command-line reference is
  `references/kdotpy-1.4.1/kdotpy/docs/helpfile.txt`. The tree is provided for reading;
  the package is NOT installed in this environment and you do not run it.
  Consult the source whenever the command-line reference leaves a question
  open.
- No internet access.

## Device

- Grown on a (001)-oriented CdTe substrate; all layers are pseudomorphically
  strained to the substrate.
- Layer sequence from the substrate up: 55 nm Hg0.3Cd0.7Te (bottom barrier),
  45 nm HgTe (well), 55 nm Hg0.3Cd0.7Te (top barrier).
- Above the top barrier sit a gate dielectric and a metal gate. They are NOT
  part of the quantum-mechanical layer stack; the gate enters the model only
  through the boundary condition and the prescribed total densities below.

## Model requirements

- Eight-orbital Kane Hamiltonian, discretized along the growth direction z;
  the in-plane momentum is a good quantum number.
- The engine's built-in HgTe / CdTe / Hg(1-x)Cd(x)Te material parameter set,
  with the following adjustments:
  - The Kane energy (squared Kane matrix element) of the well material is
    set to E_P = 20.8 eV. This adjustment must leave the band-edge effective
    masses at their established values: compensate it in the other band
    parameters rather than changing the momentum matrix element alone.
  - The valence-band offset between HgTe and CdTe is set to -620 meV. The
    band alignment of the alloy barriers must be consistent with this
    adjusted offset. Apply this as a valence-edge adjustment, retaining the
    built-in conduction-band edges.
  - Static dielectric constants: 20.8 for HgTe and 13.6 for the (Hg,Cd)Te
    barriers.
- Include the band-structure terms beyond the axial approximation: the
  topmost valence band forms anisotropic hole pockets in the in-plane
  momentum plane, and the solves below reach densities at which these
  pockets are occupied.
- The density integration must cover the occupied states: sample the
  in-plane momentum plane out to at least |k| = 1.0 nm^-1 (subband densities
  are converged by that radius). Crystal symmetry allows restricting the
  sampling to one quarter of the plane, but the anisotropy within that
  quadrant must be resolved -- a single momentum direction, or only the two
  symmetry-equivalent axis directions bounding the quadrant, is not
  sufficient.
- Zero magnetic field; zero temperature (the experiment runs at tens of mK,
  so thermal broadening is negligible).

## The three solves

One self-consistent solve per total carrier density n_tot (sign convention:
positive = net electron doping). A uniform background charge keeps the
system neutral at the charge-neutrality point.

| label | n_tot |
|---|---|
| nsym | +1.8007 x 10^11 cm^-2 |
| m2   | -2.0 x 10^11 cm^-2 |
| p2   | +2.0 x 10^11 cm^-2 |

The density is prescribed directly as the total carrier density, not through
a gate voltage.

## Electrostatic boundary condition

The device is gated from the top only. From the upstream transport analysis
you are given one experimental fact: the total density at which the two
topological-surface-state densities coincide -- the symmetric density, at
which the self-consistent Hartree potential is inversion-symmetric about the
well center -- is n_sym = 1.8007 x 10^11 cm^-2 (the nsym solve's density).

The boundary condition is identical for all three solves: hold the electric
field in the bottom barrier at its value in the symmetric configuration
defined above, and leave the top-barrier field free. Determine the fixed
bottom-barrier field from the given data.

## Self-consistency

Iterate the charge-potential loop until the Hartree potential is converged
to 0.1 meV or better.

## What to submit

Write `answer.json` at the workspace root containing exactly:

```json
{"runs": {"nsym": "answer/nsym/run.sh", "m2": "answer/m2/run.sh", "p2": "answer/p2/run.sh"}}
```

Each `run.sh` is a shell script whose payload is the full kdotpy command
line for that solve. Requirements:

- one kdotpy invocation per script, with every kdotpy argument written
  literally on the command line (no shell-variable substitution inside the
  invocation, no auxiliary parameter files);
- comment lines and environment-setup lines (exports, environment
  activation) are allowed; the grader reads only the kdotpy invocation;
- supply every numerical argument and every numerical material-parameter
  assignment as an explicit numeric literal. Signed numbers, decimals, and
  scientific notation are allowed. Do not use arithmetic expressions,
  function calls, named constants, references to other parameters, or shell
  variables as numerical values. Compute such values before writing the
  commands. Parameter names, material names, nonnumeric option keywords,
  and the tool's free-boundary marker retain their normal syntax;
- backslash line continuations are allowed;
- parallelization settings, output naming, and working-directory choices are
  free;
- each command must be runnable as written: a command the tool would reject at
  start-up (a mandatory setting missing, a value it cannot parse) is wrong.
