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

import matplotlib.colors as mplcolors
from matplotlib import colormaps as mplcolormaps

### Custom colormaps ###

# Color map bluered (blue-gray-red)
cmapdata_bluered = {
	'red':   [(0.000, 0.0, 0.0), (0.500, 0.5, 0.5), (1.000, 1.0, 1.0)],
	'green': [(0.000, 0.0, 0.0), (0.500, 0.5, 0.5), (1.000, 0.0, 0.0)],
	'blue':  [(0.000, 1.0, 1.0), (0.500, 0.5, 0.5), (1.000, 0.0, 0.0)]}
cmapdata_grayred = {
	'red':   [(0.000, 0.5, 0.5), (1.000, 1.0, 1.0)],
	'green': [(0.000, 0.5, 0.5), (1.000, 0.0, 0.0)],
	'blue':  [(0.000, 0.5, 0.5), (1.000, 0.0, 0.0)]}
cmapdata_bluered_dual = {
	'red':   [(0.000, 0.0, 0.0), (0.500, 0.4, 0.6), (1.000, 1.0, 1.0)],
	'green': [(0.000, 0.0, 0.0), (0.500, 0.4, 0.4), (1.000, 0.0, 0.0)],
	'blue':  [(0.000, 1.0, 1.0), (0.500, 0.6, 0.4), (1.000, 0.0, 0.0)]}
cm_bluered = mplcolors.LinearSegmentedColormap('bluered', cmapdata_bluered)
cm_grayred = mplcolors.LinearSegmentedColormap('grayred', cmapdata_grayred)
cm_bluered_dual = mplcolors.LinearSegmentedColormap('bluereddual', cmapdata_bluered_dual)
mplcolormaps.register(cmap=cm_bluered)
mplcolormaps.register(cmap=cm_grayred)
mplcolormaps.register(cmap=cm_bluered_dual)

# Color map yrbc (yellow-red-blue-cyan)
cmapdata_yrbc = {
	'red':   [(0.000, 0.0, 0.0), (0.333, 0.0, 0.0), (0.500, 0.5, 0.5), (0.667, 1.0, 1.0), (1.000, 1.0, 1.0)],
	'green': [(0.000, 1.0, 1.0), (0.333, 0.0, 0.0), (0.500, 0.5, 0.5), (0.667, 0.0, 0.0), (1.000, 1.0, 1.0)],
	'blue':  [(0.000, 1.0, 1.0), (0.333, 1.0, 1.0), (0.500, 0.5, 0.5), (0.667, 0.0, 0.0), (1.000, 0.0, 0.0)]}
cmapdata_yrbc2 = {
	'red':   [(0.000, 0.0, 0.0), (0.200, 0.3, 0.3), (0.333, 0.0, 0.0), (0.500, 0.5, 0.5), (0.667, 1.0, 1.0), (0.800, 0.9, 0.9), (1.000, 1.0, 1.0)],
	'green': [(0.000, 1.0, 1.0), (0.200, 0.5, 0.5), (0.333, 0.0, 0.0), (0.500, 0.5, 0.5), (0.667, 0.0, 0.0), (0.800, 0.5, 0.5), (1.000, 1.0, 1.0)],
	'blue':  [(0.000, 1.0, 1.0), (0.200, 0.9, 0.9), (0.333, 1.0, 1.0), (0.500, 0.5, 0.5), (0.667, 0.0, 0.0), (0.800, 0.3, 0.3), (1.000, 0.0, 0.0)]}
cm_yrbc = mplcolors.LinearSegmentedColormap('yrbc', cmapdata_yrbc)
cm_yrbc2 = mplcolors.LinearSegmentedColormap('yrbc2', cmapdata_yrbc2)
mplcolormaps.register(cmap=cm_yrbc)
mplcolormaps.register(cmap=cm_yrbc2)

# Color map allwhite and allgray
cmapdata_allwhite = {
	'red':   [(0.000, 1.0, 1.0), (1.000, 1.0, 1.0)],
	'green': [(0.000, 1.0, 1.0), (1.000, 1.0, 1.0)],
	'blue':  [(0.000, 1.0, 1.0), (1.000, 1.0, 1.0)]}
cmapdata_allgray = {
	'red':   [(0.000, 0.875, 0.875), (1.000, 0.875, 0.875)],
	'green': [(0.000, 0.875, 0.875), (1.000, 0.875, 0.875)],
	'blue':  [(0.000, 0.875, 0.875), (1.000, 0.875, 0.875)]}
cm_allwhite = mplcolors.LinearSegmentedColormap('allwhite', cmapdata_allwhite)
cm_allgray = mplcolors.LinearSegmentedColormap('allgray', cmapdata_allgray)
mplcolormaps.register(cmap=cm_allwhite)
mplcolormaps.register(cmap=cm_allgray)

try:
	t20map = mplcolormaps['tab20']
except KeyError:
	t20map = None
if t20map is not None:
	t20altcolors = t20map.colors[0::2] + t20map.colors[1::2]
	t20altmap = mplcolors.ListedColormap(t20altcolors, name='tab20alt')
	mplcolormaps.register(cmap=t20altmap)
	t21posnegcolors = tuple(list(t20map.colors)[-2::-2]) + ((0.0, 0.0, 0.0),) + t20map.colors[1::2]
	t21posnegmap = mplcolors.ListedColormap(t21posnegcolors, name='tab21posneg')
	mplcolormaps.register(cmap=t21posnegmap)

