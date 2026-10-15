# q2r.x Input Description - Quantum ESPRESSO v7.5

## Overview

**q2r.x** is a PHonon utility in Quantum ESPRESSO that reads force constant matrices C(q) produced by the ph.x code for a grid of q-points and calculates the corresponding set of interatomic force constants (IFC), C(R).

## Input Structure

The input follows this format:

```
&INPUT
   ...namelist variables...
/

[ nr1 nr2 nr3
  nfile
     file(1)
     file(2)
     ...
     file(nfile) ]
```

The bracketed section is conditional on whether "fildyn"0 file exists.

---

## Namelist: &INPUT

### fildyn
- **Type:** CHARACTER
- **Status:** REQUIRED
- **Description:** Input file name containing q-point grid information. Files are named "fildyn"0 (grid info) and "fildyn"1-N (force constants). For non-collinear/spin-orbit cases, files are in .xml format with .xml extension appended.

### flfrc
- **Type:** CHARACTER
- **Status:** REQUIRED
- **Description:** Output file containing the interatomic force constants in real space.

### zasr
- **Type:** CHARACTER
- **Default:** 'no'
- **Description:** Acoustic Sum Rules type for Born effective charges.
- **Allowed Values:**
  - `'no'` -- No Acoustic Sum Rules imposed (default)
  - `'simple'` -- Previous ASR implementation; diagonal correction
  - `'crystal'` -- 3 translational ASR via optimized IFC projection
  - `'one-dim'` -- 3 translational + 1 rotational ASR (periodicity axis)
  - `'zero-dim'` -- 3 translational + 3 rotational ASR
  
  Note: Not all rotational ASR apply in certain geometries; excess constraints are cancelled during orthonormalization.

### loto_2d
- **Type:** LOGICAL
- **Description:** Set to `.true.` to activate two-dimensional treatment of LO-TO splitting.

### write_lr
- **Type:** LOGICAL
- **Default:** .false.
- **Description:** Set to `.true.` to write long-range IFC into q2r file (required for asr='all' in matdyn for infrared-active solids). Adds column for text format; creates IFC_LR tag for xml format.

---

## Conditional Input (if "fildyn"0 missing)

### Line of Input: nr1 nr2 nr3
- **nr1, nr2, nr3** (INTEGER): Dimensions of the FFT grid formed by the q-point grid

### Card: fileSpecs

#### nfile
- **Type:** INTEGER
- **Description:** Number of files containing C(q_n), n=1 to nfile

#### file(1), file(2), ..., file(nfile)
- **Type:** CHARACTER
- **Description:** Names of files containing force constants. File order is flexible provided q=0 is listed first.

---

*Source: https://www.quantum-espresso.org/Doc/INPUT_Q2R.html*
