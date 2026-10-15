# Offline reference library

The agent had no internet access during the evaluation reported in the paper.
Instead it was given the software manuals in this directory. Each
subdirectory holds one code's documentation, unchanged from upstream, with
the `README.md` the agent saw beside it: origin, version, licence and what
was left out.

| Code | Documentation | Upstream | Licence | Files |
|---|---|---|---|---|
| Quantum ESPRESSO (`qe/`) | 7.5 | [www.quantum-espresso.org/Doc/INPUT_PW.html](https://www.quantum-espresso.org/Doc/INPUT_PW.html) | GPL | 5 |
| LAMMPS (`lammps/`) | upstream commit c46bbef | [github.com/lammps/lammps](https://github.com/lammps/lammps/tree/c46bbef/doc/src) | GPL-2.0 | 1,058 |
| ABACUS (`abacus/`) | upstream commit 30db3a4 | [github.com/deepmodeling/abacus-develop](https://github.com/deepmodeling/abacus-develop/tree/30db3a4/docs) | LGPL-3.0 | 64 |
| Wannier90 (`wannier90/`) | user guide and tutorial 3.1 | [www.wannier.org/user_guide.html](http://www.wannier.org/user_guide.html) | GPL-2.0-or-later | 6 |
| VAMPIRE (`vampire/`) | 7.0.0, commit 525bc27 | [github.com/richard-evans/vampire](https://github.com/richard-evans/vampire/tree/525bc27ee44c525aee229570f30f3d4c61d54f66/manual) | GPL-2.0 / BSD-2-Clause | 11 |
| ALAMODE (`alamode/`) | 1.4.2 | [github.com/alamode-code/alamode](https://github.com/alamode-code/alamode/tree/e1864d0fae37f140ab9cd2304f098ce29380fba3/docs/source) | MIT | 7 |

Files counts every file of the directory except its `README.md`. Licence
gives a short name; the full wording is in each directory's `README.md`.

## Not included

The VASP wiki pages and the LOBSTER User's Guide and FAQ were also available to
the agent but are not redistributed: VASP because the wiki's terms of use grant
no redistribution, LOBSTER because its download agreement forbids making any
part of the package accessible to third parties.
`lobster/MANUALS_NOT_INCLUDED.md` lists the LOBSTER files with sizes and sha256
so that a licensed user can restore them.

The manuals keep their upstream licences; the benchmark's CC BY 4.0 licence
does not apply to them.
