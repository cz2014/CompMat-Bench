# Task: write the kdotpy run set for a self-consistent Hartree sweep of a gated HgTe quantum well

Write three kdotpy 1.4.1 self-consistent Hartree run specifications; do not
execute them. Its source is staged at `references/kdotpy-1.4.1/`; the
package is not installed. The workspace is offline.

## Device and model

Use a $(001)$ CdTe substrate and pseudomorphically strained layers, from
bottom to top: $55\,\mathrm{nm}$ Hg0.3Cd0.7Te / $45\,\mathrm{nm}$ HgTe /
$55\,\mathrm{nm}$ Hg0.3Cd0.7Te. The top dielectric and metal gate enter only
through electrostatic boundary conditions, outside the quantum layer stack.
Use the eight-orbital Kane Hamiltonian discretized along growth direction
$z$, at zero magnetic field and temperature, with the engine's built-in
material parameters modified as follows:

- Set the well's Kane energy $E_P=20.8\,\mathrm{eV}$ while preserving its
  established band-edge effective masses.
- Set the HgTe–CdTe valence-band offset to $-620\,\mathrm{meV}$, with
  consistent alloy alignment, retaining built-in conduction-band edges.
- Use static dielectric constants 20.8 in HgTe and 13.6 in the barriers.

Include terms beyond the axial approximation. The density integration must
cover the occupied in-plane states out to at least $|k|=1.0\,\mathrm{nm}^{-1}$.
Symmetry permits restricting the domain to one quadrant, whose anisotropic
hole pockets must be resolved.

## Densities and electrostatics

Positive total density denotes electron doping. Use a uniform background
charge that makes the system neutral at charge neutrality.

| Run | Total density ($\mathrm{cm}^{-2}$) |
|---|---:|
| nsym | $+1.8007\times10^{11}$ |
| m2 | $-2.0\times10^{11}$ |
| p2 | $+2.0\times10^{11}$ |

The symmetric density is $n_{\mathrm{sym}}=1.8007\times10^{11}\,\mathrm{cm}^{-2}$:
the two surface-state densities coincide and the self-consistent Hartree
potential is inversion-symmetric about the well centre. In all three runs,
fix the bottom-barrier electric field to its value in this symmetric
configuration and leave the top-barrier field free. Determine that fixed
field from the given data. Prescribe density directly and converge the
Hartree potential to $0.1\,\mathrm{meV}$ or better.

## Submission
```json
{"runs": {"nsym": "answer/nsym/run.sh", "m2": "answer/m2/run.sh", "p2": "answer/p2/run.sh"}}
```

Write `answer.json` in the workspace root with exactly the mapping above.
Each listed `run.sh` must contain
one runnable kdotpy invocation with all arguments written literally, without
auxiliary parameter files or shell-variable substitution in the invocation.
All numerical arguments and material assignments must be explicit numeric
literals, not expressions, constants or references to other parameters.
Nonnumeric option keywords, parameter/material names and the free-boundary
marker retain their normal syntax. Comments, environment setup and backslash
continuations are allowed. Parallelization, output names and working
directories are free.
