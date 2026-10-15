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

import sys
import os
from . import config

config_help_text = \
"""Syntax:
  kdotpy config file
  kdotpy config edit
  kdotpy config list    (alias: kdotpy config show)
  kdotpy config all     (alias: kdotpy config full, kdotpy config fulllist)
  kdotpy config help <item>
  kdotpy config reset <item>
  kdotpy config set <item>=<value>
  kdotpy config <item>
  kdotpy config <item>=<value>

<item> can be any configuration item
<value> is the value that <item> is set to
You can combine multiple configuration items with semicolons and single quotes,
for example:
  kdotpy config 'fig_lmargin=10;fig_rmargin=5'
"""


#### MAIN PROGRAM ####
def main(args: list[str] | None = None) -> None:
	sourcedir = os.path.dirname(os.path.realpath(__file__))
	helpfile = os.path.join(sourcedir, 'docs', 'helpfile.txt')

	sysargv = sys.argv[2:] if args is None else args
	# Note: We use the "plain" sys.argv or args here. For kdotpy config, we
	# should not parse it with cmdargs.initialize().
	if len(sysargv) == 0:
		sys.stderr.write("ERROR (kdotpy-config.py): Additional arguments required.\n")
		sys.stdout.write(config_help_text)
		exit(1)

	config.initialize_config(sysargv=sysargv, warn_deprecated=False)

	arg1 = sysargv[0].lower()

	if arg1 == 'file':
		for f in config.get_configfiles():
			print(f)
	elif arg1 == 'edit':
		configfiles = config.get_configfiles()
		if len(configfiles) == 0:
			sys.stderr.write("ERROR (kdotpy-config.py): Configuration file does not exist.\n")
			exit(1)
		config.edit_configfile(configfiles[-1])
	elif arg1 == 'reset':
		config_keys = [key.split('=')[0] for arg in sysargv[1:] for key in arg.split(";")]
		config.check_config(config_keys)
		for key in config_keys:
			config.reset_config(key)  # invalid and deprecated keys implicitly ignored
		config.write_config(deprecate=config_keys)
	elif arg1 == 'set':
		config_keys = [key.split('=')[0] for arg in sysargv[1:] for key in arg.split(";")]
		config.parse_config(config_keys)
		config.write_config()
	elif arg1 == 'help':
		config_keys = [key.split('=')[0] for arg in sysargv[1:] for key in arg.split(";")]
		if len(config_keys) == 0:
			sys.stdout.write(config_help_text)
		else:
			config.config_help(config_keys, helpfile=helpfile)
	elif arg1 in ['list', 'show']:
		all_config = config.get_all_config()
		for key, val in all_config.items():
			print(f"{key}={val}")
	elif arg1 in ['all', 'full', 'fulllist']:
		all_config = config.get_all_config(omit_default=False)
		for key, val in all_config.items():
			print(f"{key}={val}")
	else:
		config_keys = [key for arg in sysargv for key in arg.split(";")]
		set_keys = [key for key in config_keys if '=' in key]
		get_keys = [key for key in config_keys if '=' not in key]

		if len(set_keys) > 0:
			config.parse_config(set_keys)
			config.write_config()
		config.check_config(get_keys)
		for key in config_keys:  # get_keys and set_keys
			key = key.split('=')[0]
			try:
				val = config.get_config(key)
			except KeyError:
				pass
			else:
				print(f"{key}={val}")

if __name__ == '__main__':
	main()

