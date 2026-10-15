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

import numpy as np

def trapezoidal_integration(arr):
    """Trapezoidal integration
    Yields a primitive function of the same length as the input array. The
    return value is the sum over all array elements. The total sum is conserved,
    symmetry is not.

    Argument:
    arr   Numpy array

    Return
    prim  Numpy array. A primitive function of input array arr.
    """
    zero = np.array([0.0])

    arr_r = np.concatenate((arr[1:], zero))  # f(a_{j+1})
    arr_l = np.concatenate((zero, arr[:-1]))  # f(a_{j-1})

    l_trap = 0.375 * arr + 0.125 * arr_l
    r_trap = 0.375 * arr + 0.125 * arr_r
    l_trap[-1] = l_trap[-1] + 0.125 * arr[-1]
    r_trap[0] = r_trap[0] + 0.125 * arr[0]

    return np.cumsum(l_trap) + np.cumsum(r_trap)


def basic_integration(arr):
    """Basic integration
    Yields a primitive function of the same length as the input array. The
    return value is (B[i] + B[i+1]) / 2, where B is the cumulative sum of the
    input array.

    Argument:
    arr   Numpy array

    Return
    prim  Numpy array. A primitive function of input array arr.
    """
    zero = np.array([0.0])

    arr_ext = np.concatenate((zero, arr))
    arr_cs = np.cumsum(arr_ext)

    return 0.5 * (arr_cs[1:] + arr_cs[:-1])


def integrate_arr(arr):
    """Alias for either trapeziodal_integration or basic_integration"""
    return basic_integration(arr)


def special_diff(arr, y0=None, i0=0, automatic=False):
    """Special derivative.
    Designed to be the inverse of basic_integration(). That function takes the
    (B[i] + B[i+1]) / 2, essentially a convolution. Thus, here we need to invert
    that step by doing a deconvolution. This requires the initial value as extra
    input, otherwise we get an alternating error +q, -q, +q, -q, ... . This
    function corrects this value either from explicit input or automatically, by
    assumption that the edges are linear.

    Arguments:
    arr        Numpy array of 1 dim.
    y0         Float or None. If given, the initial value.
    i0         Integer. Where the initial value should be applied.
    automatic  True or False. If True, test if function is linear at the left
               and/or right edge and apply due correction. If no linearity is
               detected, no correction is applied.

    Returns:
    diff_arr   Numpy array. The derivative of arr.
    """

    if not isinstance(arr, np.ndarray):
        raise TypeError("Argument arr must be a numpy array")
    if arr.ndim != 1:
        raise ValueError("Argument arr must be of dimension 1")
    # Do deconvolution of (B[i] + B[i+1]) / 2
    y = [0.0]
    for x1 in np.diff(arr):
        y.append(2 * x1 - y[-1])
    # Determine and apply correction term
    m = (-1) ** np.mod(np.arange(len(arr)), 2)
    if automatic and len(arr) > 3:
        ql = (y[2] - 2 * y[1] + y[0]) / 4
        qr = (y[-3] - 2 * y[-2] + y[-1]) / 4
        condl = (abs(y[2] - y[0]) < 1e-12 * abs(y[1] - y[0]))
        condr = (abs(y[-3] - y[-1]) < 1e-12 * abs(y[-2] - y[-1]))
        q = (ql + qr) / 2 if condl and condr else ql if condl else qr if condr else 0
    elif y0 is not None:
        i0 %= len(arr)  # index modulo array length
        q = (y[i0] - y0) * m[i0]
    else:
        q = 0
    return np.array(y) - q * m
