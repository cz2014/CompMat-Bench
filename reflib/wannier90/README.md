# wannier90 offline documentation

code: Wannier90
version_documented: user guide and tutorial 3.1; solution booklet 3.1.0
upstream: http://www.wannier.org/user_guide.html ; http://www.wannier.org/doc/tutorial.pdf ; http://www.wannier.org/doc/solution_booklet.pdf
licence: GPL-2.0-or-later (user guide: all material in this distribution); standalone tutorial and solution booklet contain no separate licence statement
licence_note: All the material in this distribution is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 2 of the License, or (at your option) any later version.
redistributable: yes
pulled: 2026-09-13
method: cp of all three original PDF manuals; pdftotext -layout creates same-basename UTF-8 text extracts; no PDF compression, splitting or size-based exclusion
local_origin: epw_manul/wannier90-user-guide/ ; epw_manul/wannier90-tutorial/ ; epw_manul/wannier90-tutorial-solutions/
files: 6
size: 18.067774 MB
excluded: locally authored summary.md files; duplicate PDFs in raw/

## Scope and provenance

The user guide Licence section explicitly applies GPL-2.0-or-later to all material in its distribution. That distribution-wide statement is the recorded basis for the guide, tutorial and tutorial solutions from the same 3.1 release; the latter two do not repeat the grant separately. Preserve licence terms/notices for public redistribution. These are the software authors' generic tutorials, not benchmark source-paper solutions. The complete 15,170,909-byte solution booklet is included. PDF text extracts preserve document text; no images are extracted as separate files.

Counts and sizes cover every file in this code directory except this README. MB means 1,000,000 bytes. No size ceiling applies. `pulled` records collection into this library; earlier source-mirror dates are stated separately.

## File map

- `wannier90-tutorial/`: manual chapter/source subtree
- `wannier90-tutorial-solutions/`: manual chapter/source subtree
- `wannier90-user-guide/`: manual chapter/source subtree
- `README.md`: source, licence, scope and file map
