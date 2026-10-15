# qe offline documentation

code: Quantum ESPRESSO input descriptions
version_documented: 7.5
upstream: https://www.quantum-espresso.org/Doc/INPUT_PW.html ; https://www.quantum-espresso.org/Doc/INPUT_PH.html ; https://www.quantum-espresso.org/Doc/INPUT_Q2R.html ; https://www.quantum-espresso.org/Doc/INPUT_pw2wannier90.html
licence: GNU GPL with Quantum ESPRESSO per collection brief; version and document grant not stated in the supplied input-description files
licence_note: not stated
redistributable: yes
pulled: 2026-09-13
method: cp of INPUT_PW.md, INPUT_PH.md and INPUT_Q2R.md unchanged; single named pw2wannier90 HTML fetched and converted with in-memory table-container flattening plus pandoc; no HTML saved
local_origin: epw_manul/pw-x-input/ ; epw_manul/ph-x-input/ ; epw_manul/q2r-x-input/ ; fresh pull for INPUT_PW2WANNIER90.md
files: 5
size: 0.052723 MB
excluded: locally authored summary.md files; unrelated EPW material and other input-description pages; original HTML

## Scope and provenance

The collection brief classifies the licence as GNU GPL with Quantum ESPRESSO, but no original licence sentence accompanies these local input descriptions or the newly fetched single page. Scope is the four named programs, not the entire QE manual. Existing INPUT_* Markdown is preserved as supplied rather than rewritten or verified against every upstream variable. CONVERSION.md records full visible-text preservation for the newly fetched page. On 2026-10-14 the authors ruled these files redistributable as open-software documentation.

Counts and sizes cover every file in this code directory except this README. MB means 1,000,000 bytes. No size ceiling applies. `pulled` records collection into this library; earlier source-mirror dates are stated separately.

## File map

- `CONVERSION.md`: verification record for the single QE HTML conversion
- `INPUT_PH.md`: software reference page: INPUT PH
- `INPUT_PW.md`: software reference page: INPUT PW
- `INPUT_PW2WANNIER90.md`: software reference page: INPUT PW2WANNIER90
- `INPUT_Q2R.md`: software reference page: INPUT Q2R
- `README.md`: source, licence, scope and file map
