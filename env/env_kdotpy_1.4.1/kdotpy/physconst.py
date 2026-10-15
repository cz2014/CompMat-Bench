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

### PHYSICAL CONSTANTS ###
#
# units:
# length:      nm
# energy:      meV  (CAUTION: not eV!)
# voltage:     mV   (CAUTION: not V!)
# time:        ns
# temperature: K
# magn. field: T
# charge:      e    (elementary charge; e > 0)

m_e = 0.510998910e9                     # meV (electron mass in energy equivalents; E = m c^2)
e_el = 1.6021766208e-19                 # C (elementary charge in Coulomb)
cLight = 299792458.                     # nm / ns
hbar = 6.582119514e-4                   # meV ns
hbarm0 = hbar**2 * cLight**2 / m_e / 2  # ~ 38 meV nm^2
eoverhbar = 1e-6 / hbar                 # 1 / (T nm^2) -- see note
muB = 5.7883818012e-2                   # meV / T
kB = 8.6173303e-2                       # meV / K
eovereps0 = 1.80951280207e4             # mV nm
gg = 2                                  # gyromagnetic ratio (dimensionless)
r_vonklitzing = 25812.8074555           # ohm

# Note on eoverhbar:
# The factor 1e-6 is included such that:
# eoverhbar * A (where A is vector potential in units of T nm) has a resulting
#   unit of nm^-1, as appropriate for a momentum quantity
# eoverhbar * b (where b is magnetic field / flux density in T) has a resulting
#   unit of nm^-2, so that multiplication by an area in nm^2 yields a
#   dimensionless quantity.
