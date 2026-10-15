# kdotpy - k·p theory on a lattice for simulating semiconductor band structures
# Copyright (C) 2024-2026 The kdotpy collaboration <kdotpy@uni-wuerzburg.de>
#
# SPDX-License-Identifier: GPL-3.0-only
#
# This file is part of kdotpy.
#
# kdotpy is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, version 3.
#
# kdotpy is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# kdotpy. If not, see <https://www.gnu.org/licenses/>.
#
# Under Section 7 of GPL version 3 we require you to fulfill the following
# additional terms:
#
#     - We require the preservation of the full copyright notice and the license
#       in all original files.
#
#     - We prohibit misrepresentation of the origin of the original files. To
#       obtain the original files, please visit the Git repository at
#       <https://git.physik.uni-wuerzburg.de/kdotpy/kdotpy>
#
#     - As part of a scientific environment, we believe it is reasonable to
#       expect that you follow the rules of good scientific practice when using
#       kdotpy. In particular, we expect that you credit the original authors if
#       you benefit from this program, by citing our work, following the
#       citation instructions in the file CITATION.md bundled with kdotpy.
#
#     - If you make substantial changes to kdotpy, we strongly encourage that
#       you contribute to the original project by joining our team. If you use
#       or publish a modified version of this program, you are required to mark
#       your material in a reasonable way as different from the original
#       version.

from ctypes import byref, c_int32, c_double, c_char, cdll, POINTER
import numpy as np
from scipy.sparse import csr_matrix

# region Define data types globally for feastsolver.py
mkl_int = c_int32
np_ctlib_flags = ['C_CONTIGUOUS', 'ALIGNED']
array_1d_int = np.ctypeslib.ndpointer(
    dtype=mkl_int,
    ndim=1,
    flags=np_ctlib_flags
)
array_1d_double = np.ctypeslib.ndpointer(
    dtype=c_double,
    ndim=1,
    flags=np_ctlib_flags
)
array_1d_complex = np.ctypeslib.ndpointer(
    dtype=np.complex128,
    ndim=1,
    flags=np_ctlib_flags
)
array_2d_complex = np.ctypeslib.ndpointer(
    dtype=np.complex128,
    ndim=2,
    flags=np_ctlib_flags
)
# endregion
# Load library
try:
    mkl = cdll.LoadLibrary("libfeast_rt.so")
    FEAST_VERS = 'source'
except:
    mkl = cdll.LoadLibrary("libmkl_rt.so")
    FEAST_VERS = 'intel'
# region Define functions globally for feastsolver.py
feastInit = mkl.feastinit_ if FEAST_VERS == 'source' else mkl.feastinit
feastInit.argtypes = [POINTER(array_1d_int)]
feastSolve = mkl.zfeast_hcsrev_ if FEAST_VERS == 'source' else mkl.zfeast_hcsrev
feastSolve.argtypes = [
    POINTER(c_char),  # uplo (a= U: upper, L: lower, F: full matrix)
    POINTER(mkl_int),  # n (number nonzero elements)
    POINTER(array_1d_complex),  # a (nonzero elements)
    POINTER(array_1d_int),  # ia ([row] index of nonzero elements; + last element = n+1)
    POINTER(array_1d_int),  # ja ([col] index of nonzero elements)
    POINTER(array_1d_int),  # fpm (configuration values for solver)
    POINTER(c_double),  # epsout (OUTPUT: relative error)
    POINTER(mkl_int),  # loop (OUTPUT: number of refinement loops)
    POINTER(c_double),  # emin (minimum of search interval)
    POINTER(c_double),  # emax (maximum of search interval)
    POINTER(mkl_int),  # m0 (guess for total numbers of ev)
    POINTER(array_1d_double),  # e (OUTPUT: eval, first m of m0 filled)
    POINTER(array_2d_complex),  # x (OUTPUT: evec per column, first m of m0 filled)
    POINTER(mkl_int),  # m (OUTPUT: number of found ev)
    POINTER(array_1d_double),  # res (OUTPUT: residual vector)
    POINTER(mkl_int)
    # info (OUTPUT: https://software.intel.com/content/www/us/en/develop/documentation/mkl-developer-reference-c/top/extended-eigensolver-routines/extended-eigensolver-interfaces-for-eigenvalues-within-interval/extended-eigensolver-output-details.html#extended-eigensolver-output-details_GUID-E1DB444D-B362-4DBF-A1DF-DA68F7FB7019)
]
# endregion
# region Init feast variables
fpm = np.zeros((128,), dtype=mkl_int)  # Array holds basic FEAST solver config parameters (see doc)
feastInit(fpm.ctypes.data_as(POINTER(array_1d_int)))
matType = c_char(b'F')  # specifies matrix type, use F for full matrix
epsout = c_double(-1)  # output buffer, holds eps value (relative error) after calculation finishes


# endregion


def feastsolve(mat, emin, emax, n_evals, evec=None, verbose=False, check_inputs=False):
    fpm[0] = 1 if verbose else 0  # print status
    fpm[26] = 1 if (check_inputs and FEAST_VERS == 'intel') else 0  # check inputs, only Intel MKL (FEAST4 not documented)
    # fpm[1] = 8  # N contour points
    # fpm[2] = 9  # Stopping criterion
    # fpm[15] = 2  # Use Zolotarev quadrature rules (only FEAST3+)
    # fpm[42] = 1  # Use IFEAST instead (only FEAST4+)
    # fpm[44] = 1  # IFEAST BiCGstab accuracy
    # fpm[45] = 30000  # BiSGstab max iterations
    n_evals *= 1.75  # size of subspace, use at least the amount of expected eigenvalues, best performance with 1.5-2x size
    n_evals = int(n_evals)
    mat = csr_matrix(mat, dtype=np.complex128)  # make sure matrix has correct data format
    values = mat.data
    row_ind = np.array(np.append(mat.indptr + 1, mat.nnz + 1), dtype=mkl_int)  # convert to one-based indices
    col_ind = np.array(mat.indices + 1, dtype=mkl_int)  # convert to one-based indices
    mat_dim = mkl_int(mat.shape[1])
    loops = mkl_int(0)
    emin = c_double(emin)
    emax = c_double(emax)
    evals = np.zeros((n_evals,), dtype=c_double)
    if evec is None:
        evec = np.zeros((n_evals, mat.shape[1]),
                        dtype=np.complex128)  # if one trust the documentation this should be the wrong dimension order, but only this (and reverse transpose later on, returns correctly normalized eigenvectors. Re(check) memory layouts???
        fpm[4] = 0  # use randomly generated subspace (no previous solution available)
    else:
        fpm[4] = 1  # use initial subspace (reuses previous solution)
        diff_len = evec.shape[0] - n_evals
        if diff_len == 0:
            pass  # evec sizes still match up, nothing to do
        elif diff_len > 0:
            evec = evec[:n_evals, :]  # old subspace was larger, only use first ones (contains converged evecs)
        else:
            evec = np.zeros((n_evals, mat.shape[1]),
                            dtype=np.complex128)  # we could try to append some random vectors to the old evec, but it is safer to let FEAST construct a new subspace
            fpm[4] = 0  # use randomly generated subspace (no previous solution available)
    m = mkl_int(0)
    res_vec = np.zeros((n_evals,), dtype=c_double)
    info = mkl_int(-1000)
    n_evals = mkl_int(
        n_evals)  # size of subspace, use at least the amount of expected eigenvalues, best performance with 1.5-2x size

    feastSolve(byref(matType), byref(mat_dim),
               values.ctypes.data_as(POINTER(array_1d_complex)),
               row_ind.ctypes.data_as(POINTER(array_1d_int)),
               col_ind.ctypes.data_as(POINTER(array_1d_int)),
               fpm.ctypes.data_as(POINTER(array_1d_int)),
               byref(epsout), byref(loops), byref(emin), byref(emax), byref(n_evals),
               evals.ctypes.data_as(POINTER(array_1d_double)),
               evec.ctypes.data_as(POINTER(array_2d_complex)),
               byref(m),
               res_vec.ctypes.data_as(POINTER(array_1d_double)),
               byref(info)
               )
    m = m.value
    info = info.value
    if info < 0 or info >= 100:
        raise RuntimeError("FEAST algorithm encountered error %d: %s\n" % (info, feastoutputmessage(info)))
    normalized = evec.T[..., :m]
    return evals[:m], normalized, evec, info


def feastoutputmessage(info):
    outputdict = {
        "-1": "Internal error for memory allocation.",
        "-2": "Internal error of the inner system solver. "
              "Possible reasons: not enough memory for inner linear system solver or inconsistent input.",
        "-3": "Internal error of the reduced eigenvalue solver.",
        "-4": "Matrix B is not positive definite.",  # This can not happen for a normal eigenvalue problem (B = eye)
        "0": "Successful!",
        "1": "No eigenvalue found in the search interval. "
             "Either no eigenvalues in range or range is orders of magnitude too large.",
        "2": "No Convergence (number of iteration loops > %d)." % fpm[3],
        "3": "Size of the subspace m0 is too small (m0 < m). Increase requested number of eigenvalues.",
        "4": "Successful return of only the computed subspace after call with fpm[13] = 1.",
        "200": "Problem with emin, emax (emin ≥ emax).",
        "201": "Problem with size of initial subspace m0 (m0 ≤ 0 or m0 >n).",
        "202": "Problem with size of the system n (n ≤ 0)."
    }
    if (-4 <= info <= 4) or (200 <= info <= 202):
        return outputdict["%d" % info]
    elif 100 <= abs(info) <= 163:
        return "Problem with the argument fpm[%d] to FEASTs Intel MKL interface." % info - 101 if (info > 0) else -info - 100
    else:
        return "Message not defined."
