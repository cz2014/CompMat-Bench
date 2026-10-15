# Task: a cation-disordered Hf$_{0.5}$Zr$_{0.5}$O$_2$ supercell that exposes every oxygen environment, and its relaxation deck

Ferroelectric hafnia-zirconia is a solid solution: in its polar orthorhombic
phase, Hf and Zr share one cation sublattice at random. Oxygen vacancies are
the defects that govern its switching and its leakage, and how easily an
oxygen leaves depends on which cations surround its site. The starting point
of any site-resolved defect study is therefore a supercell in which the cation
disorder is explicit and every distinct oxygen environment is represented,
relaxed at the experimental lattice. Your job is to build that cell for
$x = 0.5$, to label every oxygen site by its environment, and to write the
VASP input set that relaxes the cell. You do NOT run VASP; you only produce
the structure, the site table and the input files.

## What you are given

- `hfo2_parent.vasp` -- the orthorhombic Pca2$_1$ HfO$_2$ cell, in the axis
  setting and at the lattice parameters to be used throughout. Treat it as
  the reference that fixes the lattice, the cell orientation and the internal
  coordinates; build from it by replication and substitution. The lattice is
  held fixed in everything that follows.
- `potcar_catalog.yaml` -- the PAW potentials to use for Hf, Zr and O, with
  their cutoffs.

## The structure to build

Replicate the parent cell $2 \times 2 \times 2$ along $a$, $b$ and $c$, then
replace exactly half of the cations by Zr, on the cation sites of the parent,
so that the composition is Hf$_{0.5}$Zr$_{0.5}$O$_2$. The oxygen sublattice
and every internal coordinate stay those of the replicated parent: the cell
you submit is the unrelaxed starting structure, and relaxing it is the deck's
job.

The local cation environment of an oxygen site is defined as follows. Count
the Hf atoms and the Zr atoms whose distance from the oxygen, under periodic
boundary conditions, is at most 2.3 Å. In the parent structure every oxygen
has either four such neighbours or three, and the two kinds form two
sublattices, four-coordinated and three-coordinated. The environment of a
site is the pair (number of Hf, number of Zr) among those neighbours. At
$x = 0.5$ there are five possible environments on the four-coordinated
sublattice, from four Hf to four Zr, and four on the three-coordinated one,
from three Hf to three Zr: nine in all.

The requirement on the decoration: **every one of the nine environments must
occur at least once** in the cell. Which cation sites carry Zr is otherwise
yours to choose; any decoration that meets the requirement is acceptable.

## The site table

One row per oxygen of the submitted cell, giving its fractional coordinates
in the lattice of your POSCAR and its two environment counts. This is the
table a site-resolved defect study is indexed by, and it is checked against
the structure you submit.

## The deck

A fixed-lattice ionic relaxation of the built cell within PBE: the atomic
positions are relaxed, the lattice vectors and the cell shape are held at the
parent's, and the relaxation is converged until every residual force is below
$10^{-4}$ eV/Å. The material is a wide-gap insulator. Beyond that, every
choice in the deck is yours.

## What to submit

```text
answer.json                # {"runs": {"relax": "answer/relax"}}
answer/relax/POSCAR        # the built cell, unrelaxed
answer/relax/INCAR
answer/relax/KPOINTS
answer/relax/sites.csv     # the site table
```

`answer.json` in the workspace root must contain exactly:

```json
{"runs": {"relax": "answer/relax"}}
```

`sites.csv` starts with the header line `frac_a,frac_b,frac_c,n_Hf,n_Zr` and
continues with one row per oxygen, in any order: the three fractional
coordinates in the lattice of your POSCAR, then the two integer counts. Do
not submit a POTCAR; the potentials are fixed by the catalog and are not
graded from your files.
