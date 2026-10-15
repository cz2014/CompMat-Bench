# Reaction-front propagation in a molecular double-well chain

## 1. Task

Calculate the steady propagation speed of an HNC-to-HCN conversion front at
each of the three coupling strengths defined in Part 2. Obtain the molecular
energies needed for the chain potential from first-principles calculations.

### Method guidance

Relax the isolated HNC and HCN molecules and use climbing-image nudged elastic
band calculations to obtain the minimum-energy reaction path and activation
barrier.

Use the endpoint energy difference and barrier height to determine the
coefficients of the local potential defined in Part 2. Integrate the damped
chain equations for each coupling strength. A converted segment can be used
to initiate propagation.

Track the conversion front using the times at which successive sites cross
the local barrier and remain on the product side. Determine its steady speed
from the front motion after startup and before end effects.

## 2. Shared definitions and inputs

### Inputs

| File | Contents |
|---|---|
| `inputs/HNC.xyz` | Starting HNC geometry in XYZ format |
| `inputs/HCN.xyz` | Starting HCN geometry in XYZ format |
| `inputs/potcars/H/POTCAR` | VASP PBE PAW potential for H |
| `inputs/potcars/C/POTCAR` | VASP PBE PAW potential for C |
| `inputs/potcars/N/POTCAR` | VASP PBE PAW potential for N |

### Molecular energies and local potential

Treat HNC and HCN as isolated, neutral singlet molecules. Use VASP with PBE and
the supplied PAW potentials. Define Q = E_HNC − E_HCN from the relaxed
electronic energies, and E_b as the lowest connecting barrier measured from
HNC. Exclude zero-point and thermal corrections.

The mass-weighted coordinate u places HNC at 0 and HCN at Δu = 3.27 √amu Å;
the barrier is at u_b = 1.3 √amu Å. The local potential V is a sixth-degree
polynomial defined by

\[
V(0)=0,\quad V(\Delta u)=-Q,\quad V(u_b)=E_b,
\]

\[
V'(0)=V'(\Delta u)=V'(u_b)=0,\qquad V''(0)=V''(\Delta u).
\]

### Chain model

For a chain of N sites with open ends, the potential energy and dynamics are

\[
U=\sum_{i=1}^{N}V(u_i)+\frac{K}{2}\sum_{i=1}^{N-1}(u_{i+1}-u_i)^2,
\qquad C\ddot u_i=-\frac{\partial U}{\partial u_i}-C\eta\dot u_i.
\]

Use K = 1.0, 1.5 and 2.0 eV/(amu Å²), η = 7.5 ps⁻¹, and
C = 1.03642706 × 10⁻⁴ eV ps²/(amu Å²). With u in √amu Å and time in ps,
C supplies the mass-weighted unit conversion; no additional atomic mass is
multiplied into the equation. There is no local heat reservoir, reservoir
heat transfer, random force or external driving.

## 3. Deliverable and output definitions

Write `answer.json` as one JSON object containing the three numeric fields below.

| JSON field | Physical quantity | Unit / type |
|---|---|---|
| `v_K1_0_sites_per_ps` | Steady propagation speed at K = 1.0 eV/(amu Å²) | sites/ps / number |
| `v_K1_5_sites_per_ps` | Steady propagation speed at K = 1.5 eV/(amu Å²) | sites/ps / number |
| `v_K2_0_sites_per_ps` | Steady propagation speed at K = 2.0 eV/(amu Å²) | sites/ps / number |
