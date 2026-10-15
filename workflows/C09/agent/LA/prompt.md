# Curie temperature and its response to compression in CuInP2S6

## 1. Task

Calculate the unstrained Curie temperature of CuInP2S6 and its slope versus
layer-normal compression over 0–2%.

## 2. Shared definitions and inputs

### Inputs

| File | Contents |
|---|---|
| `inputs/start.vasp` | Starting CuInP2S6 structure in POSCAR format |
| `inputs/potential.pb` | DeepMD interatomic potential |

### System

Retain the supplied stacking and periodicity. Keep the in-plane lattice and
cell tilt fixed. Measure compression from the relaxed cell height at 0 K and
zero normal stress. At each strain, keep the cell fixed and use zero electric field.

## 3. Deliverable and output definitions

Write `answer.json` as one JSON object containing the two numeric fields below.

| JSON field | Physical quantity | Unit / type |
|---|---|---|
| `Tc_unstrained_K` | Curie temperature at 0% compression | K / number |
| `slope_K_per_percent_compression` | Slope of Tc versus compression over 0–2%; positive when compression raises Tc | K per percentage point / number |
