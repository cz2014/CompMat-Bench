# Single-page conversion verification

Source: https://www.quantum-espresso.org/Doc/INPUT_pw2wannier90.html

Fetched 2026-09-13. HTML SHA256: `29cfdab84128e9d1a28a72b737e1115c628f5896c9149c5827a69d2c4f19cd1a`.

QE nests its content in layout tables. Direct pandoc GFM conversion with raw HTML disabled dropped those tables and was rejected. Converted table containers to div containers in memory, preserving cell contents, then ran `pandoc -f html -t gfm-raw_html --wrap=none`. Every visible alphanumeric token from the HTML body survives in the plain rendering of the Markdown, with multiplicity checked. Variables, types, defaults and descriptions were retained; absent defaults were not invented. Raw HTML was never saved. No other upstream page was downloaded for this library.

The converted page contains 34 variable entries and 30 explicit default blocks. The upstream page does not state defaults for prefix, scdm_mu, scdm_sigma or atom_proj_dir. Matrix-element expressions with angle brackets are retained.
