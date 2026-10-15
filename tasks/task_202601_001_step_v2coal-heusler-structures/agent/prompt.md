# Build two ordered Heusler structures of V2CoAl

Construct the full-Heusler and inverse-Heusler orderings of V2CoAl as VASP
POSCAR files. Use a conventional cubic cell with lattice constant
$a = 5.95\,\text{Å}$ and the following sublattice occupations.

## (1) Full-Heusler L2_1 ordering

Space group $Fm\bar{3}m$ (No. 225):

- V on Wyckoff 8c, represented by $(1/4,1/4,1/4)$.
- Co on Wyckoff 4b, represented by $(1/2,1/2,1/2)$.
- Al on Wyckoff 4a, represented by $(0,0,0)$.

## (2) Inverse-Heusler XA ordering

Space group $F\bar{4}3m$ (No. 216):

- V on Wyckoff 4b, represented by $(1/2,1/2,1/2)$.
- V on Wyckoff 4d, represented by $(3/4,3/4,3/4)$.
- Co on Wyckoff 4c, represented by $(1/4,1/4,1/4)$.
- Al on Wyckoff 4a, represented by $(0,0,0)$.

The representative coordinates are fractional coordinates of the conventional cell.

## Output

Write `POSCAR_L21` and `POSCAR_XA` in the workspace root. Each file must contain
the complete conventional cell for its ordering. Direct or Cartesian coordinates
are accepted.
