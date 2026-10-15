# kdotpy 1.4.1 (source bundle)

Unmodified source of kdotpy 1.4.1, a pure-Python code for k.p band-structure
calculations. Upstream Git repository:
<https://git.physik.uni-wuerzburg.de/kdotpy/kdotpy>. Taken from the PyPI wheel
`kdotpy-1.4.1-py3-none-any.whl`, sha256
`9b1ff2a94babc06ad2df8849abc8fa60fbbe1bf847a48d306c04742d29e5d5d2`.

## What was omitted

Relative to the wheel, this bundle omits:

- all `__pycache__/` directories,
- `kdotpy/testselfcon.py`,
- the upstream `kdotpy-1.4.1.dist-info/`, replaced by a minimal stub that
  carries only the metadata version, name, version and license expression.

No remaining file was altered. All license headers are preserved byte for byte.

## License

GPL-3.0-only. The full text is in `LICENSE`. The additional terms that the
authors attach under Section 7 of the GPL are in `LICENSE.additional`; they
require preserving the copyright notices and license headers in the original
files, and marking any modified version as different from the original.

Citation: kdotpy: SciPost Phys. Codebases 47 (2025), DOI
10.21468/SciPostPhysCodeb.47

## Requirements and running

Python >= 3.11, numpy >= 2.0, scipy >= 1.13, matplotlib >= 3.9,
threadpoolctl >= 3.6. Run without installing:

    PYTHONPATH=<this directory> python -m kdotpy <mode> ...

The command-line reference is `kdotpy/docs/helpfile.txt`, which is also
printed by `python -m kdotpy help`.
