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

import os
import sys
import shutil
from difflib import get_close_matches
import tempfile
import re
import subprocess as subp
from typing import Protocol, Any, Iterator, TextIO, Iterable


class SysArgv(Protocol):
	def __contains__(self, item: Any) -> bool: ...
	def __getitem__(self, i: int) -> str: ...
	def __len__(self) -> int: ...
	def __iter__(self) -> Iterator: ...


### INITIAL CONFIGURATION ###

configpath = os.path.join(os.path.expanduser('~'), '.kdotpy')
configpath_legacy = os.path.join(os.path.expanduser('~'), '.hgmnte')
configpaths = [configpath, configpath_legacy]
default_config: dict[str, str] = {
	'diag_solver': 'auto',
	'diag_solver_worker_type': 'auto',
	'diag_solver_cupy_dtype': 'double',
	'diag_solver_cupy_iterations': '5',
	'diag_solver_cupy_gemm_dim_thr': '4e6',
	'diag_solver_slepc_options': '',
	'diag_save_binary_ddp': 'false',
	'diag_save_binary_ddp_delete_eivec': 'false',
	'err_unexpected_ignore': 'true',
	'err_unexpected_print_traceback': 'true',
	'task_retries': '2',
	'tasks_grouped': 'false',
	'numpy_printprecision': '6',
	'numpy_linewidth': '200',
	'job_monitor_limit': '101',
	'material_file_filter': '',
	'band_align_exp': '4',
	'band_align_ndelta_weight': '20',
	'band_char_node_threshold': '1e-6',
	'band_char_orbital_threshold': '5e-3',
	'band_char_use_minmax': 'true',
	'band_char_make_real': 'false',
	'bandindices_adiabatic_debug': 'false',
	'band_realign_derivatives': 'recalculate',
	'batch_stderr_extension': 'txt',
	'batch_stdout_extension': 'txt',
	'batch_float_format': '%s',
	'berry_dk': '1e-3',
	'berry_ll_simulate': 'false',
	'cnp_legacy_method': 'false',
	'color_dos': 'Blues',
	'color_idos': 'RdBu_r',
	'color_localdos': 'cividis,jet',
	'color_trans': 'hot_r',
	'color_potential_yz': 'viridis',
	'color_bindex': 'tab21posneg',
	'color_indexed': 'tab20alt,tab20',
	'color_indexedpm': 'tab20',
	'color_shadedpm': 'bluereddual',
	'color_ipr': 'inferno_r',
	'color_energy': 'jet',
	'color_posobs': 'grayred',
	'color_sigma': 'inferno_r',
	'color_symmobs': 'bluered',
	'color_threehalves': 'yrbc',
	'color_wf_zy': 'Blues',
	'csv_style': 'csv',
	'csv_multi_index': 'tuple',
	'csv_bandlabel_position': 'top',
	'fig_matplotlib_style': 'kdotpy.mplstyle',
	'fig_hsize': '150',
	'fig_vsize': '100',
	'fig_lmargin': '20',
	'fig_rmargin': '4',
	'fig_bmargin': '12',
	'fig_tmargin': '3',
	'fig_charlabel_space': '0.8',
	'fig_colorbar_space': '30',
	'fig_colorbar_size': '4',
	'fig_colorbar_margin': '7.5',
	'fig_colorbar_method': 'insert',
	'fig_colorbar_abstwosided': 'true',
	'fig_colorbar_labelpos': 'center',
	'fig_extend_xaxis': '0.05',
	'fig_inset_size': '30',
	'fig_inset_margin': '3',
	'fig_inset_color_resolution': '20',
	'fig_legend_fontsize': 'auto',
	'fig_spin_arrow_length': '5',
	'fig_max_arrows': '20',
	'fig_arrow_color_2d': '#c0c0c0',
	'fig_ticks_major': 'auto',
	'fig_ticks_minor': 'none',
	'fig_unit_format': '[]',
	'densityyz_save_binary': 'false',
	'dos_interpolation_points': '100',
	'dos_energy_points': '1000',
	'dos_convolution_points': '2000',
	'dos_density_axis_points': '400',
	'dos_intobs_volume_elements': 'false',
	'dos_local_use_byband': 'true',
	'dos_local_save_binary': 'false',
	'dos_local_density_at_energy_points': '0',
	'dos_print_validity_range': 'true',
	'dos_print_momentum_multiplier': 'false',
	'dos_quantity': 'p',
	'dos_unit': 'nm',
	'dos_strategy_no_e0': 'dos',
	'bhz_allow_intermediate_bands': 'false',
	'bhz_points': '200',
	'bhz_gfactor': 'false',
	'bhz_abcdm': 'false',
	'bhz_ktilde': 'true',
	'bhz_plotcolor': 'red,blue,black',
	'bhz_plotstyle': 'dotted',
	'lattice_regularization': 'false',
	'lattice_zres_strict': 'true',
	'lattice_ycoord_midpoints': 'false',
	'magn_epsilon': '-1e-4',
	'selfcon_full_diag': 'true',
	'selfcon_acceptable_status': '1',
	'selfcon_check_chaos_steps': '4',
	'selfcon_check_orbit_steps': '4',
	'selfcon_convergent_steps': '5',
	'selfcon_debug': 'false',
	'selfcon_diff_norm': 'rms',
	'selfcon_dynamic_time_step': 'false',
	'selfcon_erange_from_eivals': 'false',
	'selfcon_ll_use_broadening': 'false',
	'selfcon_energy_points': '1000',
	'selfcon_min_time_step': '0.001',
	'selfcon_potential_average_zero': 'true',
	'selfcon_symmetrization_constraint': 'strict',
	'selfcon_use_init_density': 'false',
	'transitions_min_amplitude': '0.01',
	'transitions_min_deltae': '0.1',
	'transitions_max_deltae': '0',
	'transitions_dispersion_num': '4',
	'transitions_broadening_type': 'lorentzian',
	'transitions_broadening_scale': '2.5',
	'transitions_all_if_filtered': 'false',
	'transitions_spectra': 'false',
	'transitions_plot': 'true',
	'plot_enable': 'true',
	'plot_constdens_color': 'blue',
	'plot_dispersion_default_color': 'blue',
	'plot_dispersion_energies': 'true',
	'plot_dispersion_energies_color': 'black',
	'plot_dispersion_parameter_text': 'true',
	'plot_dispersion_stack_by_index': 'false',
	'plot_dos_color': 'blue',
	'plot_dos_energies': 'true',
	'plot_dos_vertical': 'true',
	'plot_dos_validity_range': 'true',
	'plot_dos_fill': 'false',
	'plot_idos_fill': 'false',
	'plot_dos_units_negexp': 'false',
	'plot_ecnp': 'false',
	'plot_potential_yz_contours': 'true',
	'plot_potential_yz_equal_aspect': 'false',
	'plot_rasterize_pcolormesh': 'true',
	'plot_rxy_hall_slope': 'true',
	'plot_sdh_markers': 'true',
	'plot_sdh_markers_color': 'red',
	'plot_sdh_scale_amount': '0',
	'plot_transitions_labels': 'true',
	'plot_transitions_quantity': 'rate',
	'plot_transitions_max_absorption': '0.05',
	'plot_transitions_frequency_ticks': 'true',
	'plot_wf_orbitals_realshift': 'false',
	'plot_wf_orbitals_order': 'standard',
	'plot_wf_mat_label_rot': '0',
	'plot_wf_mat_min_thick_label': '0.15',
	'plot_wf_zy_format': 'pdf',
	'plot_wf_zy_bandcolors': 'hsl',
	'plot_wf_zy_scale': 'separate',
	'plot_wf_y_scale': 'size',
	'plot_wf_delete_png': 'true',
	'plot_wf_together_num': '12',
	'table_berry_precision': '4',
	'table_dos_local_files': 'csv',
	'table_dos_precision': '8',
	'table_dos_scaling': 'false',
	'table_dos_units_negexp': 'false',
	'table_densityyz_scaling': 'false',
	'table_data_label_style': 'plain',
	'table_data_unit_style': 'plain',
	'table_dispersion_precision': '5',
	'table_dispersion_data_label': 'true',
	'table_dispersion_units': 'true',
	'table_dispersion_unit_style': 'plain',
	'table_dispersion_obs_style': 'raw',
	'table_qz_precision': '5',
	'table_extrema_precision': '5',
	'table_absorption_precision': '5',
	'table_transitions_precision': '6',
	'table_transitions_deltall': 'false',
	'table_wf_files': 'csv',
	'table_wf_precision': '5',
	'table_bdependence_filter_erange': 'false',
	'table_dispersion_filter_erange': 'false',
	'wf_locations_exact_match': 'true',
	'wf_locations_filename': 'true',
	'xml_omit_default_config_values': 'false',
	'xml_shorten_command': 'false',
}
_config: dict[str, str] = {}

# Deprecated configuration keys. If these are included in a configuration file
# or on the command line, a warning will be shown.
deprecated_config: list[str] = ['table_transitions_ratecoeff_unit']

def initialize_config(
		sysargv: list[str] | SysArgv | None = None,
		reset: bool = True,
		do_write: bool = True,
		warn_deprecated: bool = True) -> None:
	"""Initialize configuration

	Recipe:
	Check legacy config (hgmnterc)
	Check whether kdotpyrc exists and if not, write it with default values
	Read kdotpyrc
	Write new config file (optionally, see argument do_write)
	Load custom configuration from command line (file or values)

	Arguments:
	sysargv   List of str, SysArgv instance or None. The command line arguments.
	          If None, the configuration is initialized with sys.argv.
	reset     True or False. Whether to clear the global configuration before
	          initialization. Setting this to True (default) is strongly
	          recommended, as otherwise the initialization step only updates
	          the configuration from the previous run. If set to False, one
	          should also set do_write to False.
	do_write  True or False. If True (default), write a new complete config file
	          after all values have been set; this should be done if this
	          function is called initially from kdotpy-xx.py. Use False when
	          this function is called for re-initialization, e.g., in a parallel
	          worker (no longer applies as of kdotpy v1.4.0).
	warn_deprecated  True or False. Whether to show a deprecation warning for
	                 deprecated configuration keys. This should be False when
	                 this function is called from kdotpy-config.py.

	No return value.
	"""
	global _config
	if reset:
		_config.clear()
	check_legacy_config()
	if not os.path.exists(os.path.join(configpath, 'kdotpyrc')):
		sys.stderr.write("Warning (initialize_config): Write new default config file\n")
	else:
		# Initial config file
		try:
			read_config(os.path.join(configpath, 'kdotpyrc'), init=True, warn_deprecated=warn_deprecated)
		except:
			sys.stderr.write("ERROR (initialize_config): Cannot read default config file\n")
			raise
	# Rewrite config file after initialization, so that possibly new keys are
	# written to the file. It is not necessary to call write_config() upon exit.
	if do_write:
		if not reset:
			sys.stderr.write("Warning (initialize_config): Writing or rewriting config file with configuration that has not been reset from a possible previous run.\n")
		write_config()
	# Load a custom config file or take values from command line
	cmdargs_config(sysargv)
	return

def check_legacy_config() -> None:
	"""Check for legacy config file hgmnterc
	Check whether the config files kdotpyrc (new) and hgmnterc (old) is present.
	If the old one is present and the new one is not, copy and rename.

	No return value.
	"""
	newcfg = os.path.join(configpath, 'kdotpyrc')
	oldcfg = os.path.join(configpath_legacy, 'hgmnterc')
	if not os.path.exists(newcfg) and os.path.exists(oldcfg):
		if not os.path.exists(configpath):
			try:
				os.mkdir(configpath)
			except:
				sys.stderr.write("ERROR (check_legacy_config): Cannot create config path\n")
				raise
		try:
			shutil.copy2(oldcfg, newcfg)
		except:
			sys.stderr.write("Warning (check_legacy_config): Copying of legacy config file to new location ('%s' -> '%s') has failed. Please try it manually.\n" % (oldcfg, newcfg))
		else:
			sys.stderr.write("Warning (check_legacy_config): Copying of legacy config file to new location ('%s' -> '%s') was successful. Please remove the old file manually.\n" % (oldcfg, newcfg))
	if os.path.exists(configpath_legacy):
		ls = os.listdir(configpath_legacy)
		n_otherfiles = len(ls)
		if 'hgmnterc' in ls:
			n_otherfiles -= 1
		if n_otherfiles > 0:
			sys.stderr.write("Warning (check_legacy_config): Legacy config path '%s' contains other files than the default config file. Please check and move config files to '%s' manually.\n" % (configpath_legacy, configpath))
		elif n_otherfiles == 0:
			sys.stderr.write("Warning (check_legacy_config): Legacy config path '%s' exists but does not contain other files than the default config file. Please remove the directory '%s' manually.\n" % (configpath_legacy, configpath_legacy))
	return

### RETRIEVAL OF CONFIG VALUES ###

def get_config(key: str, choices: list[str] | None = None, case_sensitive: bool = False, allow_default: bool = True) -> str:
	"""Get (string) configuration value.

	If the configuration key is invalid, raise KeyError

	Arguments:
	key      String. Configuration key.
	choices  None or list. If set, raise an error or warning if the
	         configuration value is not an element of the list.
	case_sensitive  False or True. If True, the 'choice' test is done in a case
	                sensitive manner. If False (default), this check is case
	                insensitive.
	allow_default   False or True. If True, a failed 'choice' test will return
	                the default value and raise a warning. If False, raise an
	                error on failed 'choice' test and exit the program.

	Returns:
	string value
	"""
	if key in _config:
		val = _config[key]
	elif key in default_config:
		val = default_config[key]
	else:
		raise KeyError(f"Invalid configuration option {key}")
	if choices is None:
		return val
	else:
		val1 = val if case_sensitive else val.lower()
		if val1 in choices:
			return val
		elif allow_default and key in default_config:
			sys.stderr.write("ERROR (get_config): Invalid value '%s' for configuration option %s. Possible values are: %s. Using default value '%s'.\n" % (val, key, ", ".join(choices), default_config[key]))
			return default_config[key]
		else:
			sys.stderr.write("ERROR (get_config): Invalid value '%s' for configuration option %s. Possible values are: %s\n" % (val, key, ", ".join(choices)))
			exit(1)

def get_config_num(key: str, minval: float | None = None, maxval: float | None = None) -> float:
	"""Get numeric configuration value

	Arguments:
	key      String. Configuration key.
	minval   None or a number. If set, the lower bound. If the actual value is
	         < minval, return minval.
	maxval   None or a number. If set, the upper bound. If the actual value is
	         > maxval, return maxval.

	Returns:
	Number of type float.
	"""
	val = get_config(key)
	try:
		val = float(val)
	except:
		sys.stderr.write("ERROR (get_config_num): Configuration option %s must be a numerical value.\n" % key)
		exit(1)
	if minval is not None and val < minval:
		sys.stderr.write("Warning (get_config_num): Configuration option %s must be a numerical value >= %s.\n" % (key, minval))
		val = minval
	if maxval is not None and val > maxval:
		sys.stderr.write("Warning (get_config_num): Configuration option %s must be a numerical value <= %s.\n" % (key, maxval))
		val = maxval
	return val

def get_config_num_auto(key: str, automatic: Iterable[str] = ('none', 'auto', 'automatic'), minval: float | None = None, maxval: float | None = None) -> float | None:
	"""Get numeric configuration value or None if the value is set to 'automatic'.

	Arguments:
	key        String. Configuration key.
	automatic  List or tuple of strings that evaluate to 'automatic'.
	minval     None or a number. If set, the lower bound. If the actual value is
	           < minval, return minval.
	maxval     None or a number. If set, the upper bound. If the actual value is
	           > maxval, return maxval.

	Development note:
	The default value of argument automatic is not modified, hence safe.

	Returns:
	None or number of type float. Value None means 'automatic'.
	"""
	val = get_config(key)
	if len(val) == 0 or val.lower() in automatic:
		return None
	try:
		val = float(val)
	except:
		sys.stderr.write("ERROR (get_config_num_auto): Configuration option %s must be a numerical value, or one of: %s.\n" % (key, ", ".join(automatic)))
		exit(1)
	if minval is not None and val < minval:
		sys.stderr.write("Warning (get_config_num_auto): Configuration option %s must be a numerical value >= %s, or one of %s.\n" % (key, minval, ", ".join(automatic)))
		val = minval
	if maxval is not None and val > maxval:
		sys.stderr.write("Warning (get_config_num_auto): Configuration option %s must be a numerical value <= %s, or one of %s.\n" % (key, maxval, ", ".join(automatic)))
		val = maxval
	return val

def get_config_int(key: str, minval: int | None = None, maxval: int | None = None) -> int:
	"""Get integer numeric configuration value

	Arguments:
	key      String. Configuration key.
	minval   None or a number. If set, the lower bound. If the actual value is
	         < minval, return minval.
	maxval   None or a number. If set, the upper bound. If the actual value is
	         > maxval, return maxval.

	Returns:
	Number of type int.
	"""
	val = get_config(key)
	try:
		val = int(val)
	except:
		sys.stderr.write("ERROR (get_config_int): Configuration option %s must be an integer value.\n" % key)
		exit(1)
	if minval is not None and val < minval:
		sys.stderr.write("Warning (get_config_int): Configuration option %s must be an integer value >= %s.\n" % (key, minval))
		val = minval
	if maxval is not None and val > maxval:
		sys.stderr.write("Warning (get_config_int): Configuration option %s must be an integer value <= %s.\n" % (key, maxval))
		val = maxval
	return val

def get_config_bool(key: str) -> bool:
	"""Get boolean configuration value

	Arguments:
	key      String. Configuration key.

	Returns:
	True or False.
	"""
	val = get_config(key)
	if val.lower() in ['yes', 'y', 'true', 't', '1', 'enabled', 'on']:
		return True
	elif val.lower() in ['no', 'n', 'false', 'f', '0', 'disabled', 'off']:
		return False
	else:
		sys.stderr.write("ERROR (get_config_bool): Configuration option %s must be a boolean value.\n" % key)
		exit(1)

def get_all_config(omit_default: bool = True) -> dict[str, str]:
	"""Get all configuration values.

	Argument:
	omit_default  False or True (default). If True, exclude all key-value pairs
	              that are set to their default values.

	Returns:
	A dict instance with all (non-default, if applicable) key-value pairs
	"""
	all_config = {}
	for key in sorted(default_config):

		if key in _config:
			if (not omit_default) or _config[key] != default_config[key]:
				all_config[key] = _config[key]
		elif not omit_default:
			all_config[key] = default_config[key]
	return all_config

def set_config(key: str, val: str) -> bool:
	"""Set configuration value"""
	if key in default_config:
		_config[key] = val
		return True
	else:
		return False

def set_all_config(all_config: dict[str, str]):
	"""Set multiple configuration values"""
	global _config
	_config.update({key: val for key, val in all_config.items() if key in default_config})
	# TODO: Add some guards, for example only allow if _config is empty, or do not allow updating existing keys
	# TODO: Optionally reset, i.e., use _config.clear() before _config.update(...)

def reset_config(key: str) -> bool:
	"""Reset configuration value to default value by deleting it from _config"""
	if key in default_config:
		if key in _config:
			del _config[key]
		return True
	else:
		return False

def config_help(keys: str | list[str], helpfile: str = 'README', suggest: bool = True) -> None:
	"""Show help for configuration values from the help file (formerly README)

	Find lines in the help file that start with a matching configuration key,
	then print to stdout until a non-indented or empty line is found.

	Arguments:
	keys         String or list. If a string, match that one key. If a list,
	             match all keys in the list.
	helpfile     String. The path to the help file (README file).
	suggest      True or False. If True, show help also for suggested
	             alternatives if there are any invalid keys in the input.

	No return value.
	"""
	if isinstance(keys, str):
		keys = [keys]
	if not isinstance(keys, list):
		raise TypeError("Argument keys must be a str or a list instance.")
	valid_keys = [key for key in keys if key in default_config]
	invalid_keys = [key for key in keys if key not in default_config and key not in deprecated_config]
	# Deprecated keys are also considered valid keys in this context, because
	# they still have an entry in the help file.
	if len(invalid_keys) > 0:
		sys.stderr.write("Warning (config_help): The input contains the following invalid key%s: %s.\n" % ("s" if len(invalid_keys) >= 2 else "", ", ".join(invalid_keys)))
		suggestions = suggest_keys(invalid_keys)
		if suggest and len(suggestions) > 0:
			sys.stderr.write("Warning (config_help): Suggested valid keys: " + ", ".join(suggestions) + ".\n")
		valid_keys.extend(suggestions)
	if len(valid_keys) == 0:
		return

	key_found = False
	pattern = re.compile(r'([A-Za-z_0-9]+)(,\s*[A-Za-z_0-9]+)*')
	with open(helpfile, 'r') as f:
		for ln in f:
			l = ln.rstrip()
			if len(l) == 0:
				key_found = False
			elif l.startswith(' ') or l.startswith('\t'):
				if key_found:
					print(l)
			elif pattern.match(l) is not None and any(key in l for key in valid_keys):
				key_found = True
				print(l)
			else:
				key_found = False

### CONFIG FILE I/O ###

def suggest_keys(invalid_keys: list[str]) -> list[str]:
	all_suggestions = []
	for key in invalid_keys:
		for suggestion in get_close_matches(key, default_config.keys(), n = 3, cutoff = 0.7):
			if suggestion not in all_suggestions:
				all_suggestions.append(suggestion)
	return all_suggestions

def check_config(keys: str | list[str], suggest: bool = True) -> bool:
	"""Check if keys are valid config values"""
	if isinstance(keys, str):
		keys = [keys]
	if not isinstance(keys, list):
		raise TypeError("Argument keys must be a str or a list instance.")
	invalid_keys = [key for key in keys if key not in default_config and key not in deprecated_config]
	if len(invalid_keys) > 0:
		sys.stderr.write("Warning (check_config): The config contains the following invalid key%s: %s.\n" % ("s" if len(invalid_keys) >= 2 else "", ", ".join(invalid_keys)))
		suggestions = suggest_keys(invalid_keys)
		if suggest and len(suggestions) > 0:
			sys.stderr.write("Warning (check_config): Suggested valid keys: " + ", ".join(suggestions) + ".\n")
	return len(invalid_keys) == 0

def cmdargs_config(sysargv: list[str] | SysArgv | None = None) -> bool:
	"""Take from command line, either a file name or a string with configuration values.
	Multiple inputs are possible.

	Argument:
	sysargv   List of str, SysArgv instance or None. The command line arguments.
	          If None, the configuration is initialized with sys.argv, where the
	          first two elements (for example 'kdotpy', '2d') are discarded.

	Returns:
	success   True or False. Whether at least one valid configuration value was
	          found.
	"""
	if sysargv is None:
		sysargv = sys.argv[2:]
	if 'config' not in sysargv:
		return False
	success = False
	for argn, arg in enumerate(sysargv):
		if arg.lower() == 'config':
			if argn + 1 >= len(sysargv):
				sys.stderr.write("ERROR (initialize_config): Argument 'config' must be followed by a valid file name or configuration values.\n")
				exit(1)
			custom_config = sysargv[argn + 1]
			if custom_config == 'default':
				sys.stderr.write("Warning (initialize_config): Using default config. Ignoring kdotpyrc and previously set values.\n")
				_config.clear()
				success = True
			elif os.path.isfile(custom_config):
				success |= read_config(custom_config)  # read file
			elif os.path.isfile(os.path.join(configpath, custom_config)):
				success |= read_config(os.path.join(configpath, custom_config))
			else:
				config_data = custom_config.split(";")  # take from command line
				success |= parse_config(config_data)
				if os.getcwd() not in configpaths:
					configpaths.append(os.getcwd())
	return True

def read_config(filename: str, init: bool = False, warn_deprecated: bool = True) -> bool:
	"""Open and read config file

	Arguments:
	filename         Filename
	init             True if file has to be interpreted as 'initial'
	                 configuration file, i.e., kdotpyrc on the default location.
	                 False if not.
	warn_deprecated  True or False. Whether to show a deprecation warning for
	                 deprecated configuration keys.

	Returns:
	True on success, False on error
	"""
	error_str = "(default)" if init else filename
	try:
		f = open(filename, 'r')
	except:
		sys.stderr.write("ERROR (read_config): Cannot read config file %s\n" % error_str)
		raise
	success = parse_config(f, error_str=error_str, warn_deprecated=warn_deprecated)
	f.close()
	filedir = os.path.dirname(os.path.abspath(filename))
	if success and filedir not in configpaths:
		configpaths.append(filedir)
	return success

def parse_config(data_or_file: list[str] | TextIO, error_str: str | None = None, warn_deprecated: bool = True) -> bool:
	"""Parse configuration from command line input.
	This function works by virtue of a generic iterable argument (data_or_file).
	When iterated over this argument, it yields strings of the form 'key=value',
	which are then parsed to configuration key-value pairs.

	Arguments:
	data_or_file   An iterable object which yields strings. This may be a list
	               of strings of the form 'key=value' or a file handler (from
	               open(filename, 'r'), for example), among others.
	error_str      None or string. This string will be used in error messages.
	               It should be the filename of the config file. If None, do not
	               print the filename in the error messages.
	warn_deprecated  True or False. Whether to show a deprecation warning for
	                 deprecated configuration keys.

	Returns:
	True on success, False on error
	"""

	global _config
	invalid_keys = []
	deprecated_keys = []
	valid_lines = 0
	for l in data_or_file:
		m = re.match(r'\s*([_a-z0-9]*)(\s*)=(\s*)(.*)', l.strip())
		if m is not None:
			key, sleft, sright, val = m.groups()
			if len(key) == 0:
				pass
			elif key in default_config:
				_config[key] = str(val)
				valid_lines += 1
			elif key in deprecated_config:
				deprecated_keys.append(key)
			else:
				invalid_keys.append(key)

	success = True
	if valid_lines == 0 and error_str != "(default)":  # ignore 'no valid keys' warning for default configuration file
		sys.stderr.write("Warning (parse_config): No valid configuration keys. Check whether %s is a valid configuration %s.\n" % ("this" if error_str is None else error_str, "input" if error_str is None else "file"))
		success = False
	elif len(invalid_keys) > 0:
		sys.stderr.write("Warning (parse_config): The config%s contains the following invalid key%s, which %s been ignored: %s.\n" % ("" if error_str is None else " file " + error_str, "s" if len(invalid_keys) >= 2 else "", "have" if len(invalid_keys) >= 2 else "has", ", ".join(invalid_keys)))
	if len(invalid_keys) > 0:
		suggestions = suggest_keys(invalid_keys)
		if len(suggestions) > 0:
			sys.stderr.write("Warning (parse_config): Suggested valid keys: " + ", ".join(suggestions) + ".\n")
	if len(deprecated_keys) > 0:
		config_str = "config" if error_str is None else f"config file {error_str}"
		key_str = "keys" if len(deprecated_keys) >= 2 else "key"
		has_str = "have" if len(deprecated_keys) >= 2 else "has"
		list_str = ", ".join(deprecated_keys)
		sys.stderr.write(f"Warning (parse_config): The {config_str} contains the following deprecated {key_str}, which {has_str} been ignored: {list_str}.\n")
		if warn_deprecated and error_str == "(default)":
			sys.stderr.write(f"Warning (parse_config): You may disable the {key_str} by using the following command:\n")
			sys.stderr.write("  kdotpy config reset " + " ".join(deprecated_keys) + "\n")
		elif warn_deprecated and error_str is not None:
			sys.stderr.write(f"Warning (parse_config): Remove the {key_str} from the config file {error_str} manually to silence this warning.\n")

	return success

def write_config(deprecate: list[str] | None = None) -> None:
	"""Write configuration file.

	Argument:
	deprecate   None or a list. Deprecate all keys in the list.

	Note:
	For kdotpy, it is not necessary to call this function upon exit, because the
	program does not change the configuration by itself.
	"""
	global _config
	if not os.path.exists(configpath):
		try:
			os.mkdir(configpath)
		except:
			sys.stderr.write("ERROR (write_config): Cannot create config file\n")
			raise

	# Read existing config file so that it can be rewritten with only the
	# relevant config keys changed, keeping the rest
	config_filename = os.path.join(configpath, 'kdotpyrc')
	config_data = []
	valid_keys = []
	if deprecate is None:
		deprecate = []
	deprecated_keys = []

	if os.path.exists(config_filename):
		with open(config_filename, 'r') as f1:
			for l in f1:
				m = re.match(r'#?\s*([_a-z0-9]*)(\s*)=(\s*)(.*)', l.strip())
				if m is None:
					config_data.append(l)
					continue
				key, sleft, sright, val = m.groups()
				if key in _config:
					valid_keys.append(key)
					val = _config[key]
					config_data.append(f"{key}{sleft}={sright}{val}\n")
				elif key in default_config:
					valid_keys.append(key)
					val = default_config[key]
					config_data.append(f"# {key}{sleft}={sright}{val}\n")
				elif key in deprecate and key in deprecated_config:
					deprecated_keys.append(key)
					if not val.endswith("# DEPRECATED"):
						val += "  # DEPRECATED"
					config_data.append(f"# {key}{sleft}={sright}{val}\n")
				else:
					config_data.append(l)

	for key in sorted(default_config):
		if key in valid_keys:
			pass
		elif key in _config:
			config_data.append(key + '=' + _config[key] + '\n')
		else:
			config_data.append('# ' + key + '=' + default_config[key] + '\n')

	if len(deprecated_keys) > 0:
		key_str = "keys" if len(deprecated_keys) >= 2 else "key"
		has_str = "have" if len(deprecated_keys) >= 2 else "has"
		list_str = ", ".join(deprecated_keys)
		sys.stderr.write(f"Warning (write_config): The following {key_str} {has_str} been commented out and marked deprecated: {list_str}.\n")

	try:
		f = tempfile.NamedTemporaryFile('w', dir = configpath, prefix ='kdotpyrc-', delete = False)
	except:
		sys.stderr.write("ERROR (write_config): Cannot create config file\n")
		raise
	tmpname = f.name
	if len(config_data) == 0 or config_data[0].strip() != '## kdotpy configuration file':
		f.write('## kdotpy configuration file\n')
	for l in config_data:
		f.write(l)
	f.close()
	try:
		os.replace(tmpname, os.path.join(configpath, 'kdotpyrc'))
	except:
		sys.stderr.write("Warning (write_config): Cannot replace config file. Can be caused by many multiple processes accessing the file simultaneously.\n")

def get_editor(default: str = 'nano') -> str:
	"""Get command for editor from environment variable"""
	if 'VISUAL' in os.environ:
		return os.environ['VISUAL']
	elif 'EDITOR' in os.environ:
		return os.environ['EDITOR']
	return default

def edit_configfile(filename: str | None = None) -> None:
	"""Edit config file using editor"""
	editor = get_editor()
	if filename is None:
		filename = os.path.join(configpath, 'kdotpyrc')
	try:
		subp.run([editor, filename])
	except OSError:
		sys.stderr.write("ERROR (edit_configfile): Unable to open editor.\n")
		raise
	else:
		print("{} {}".format(editor, filename))
	return

def get_configfiles() -> list[str]:
	global configpaths
	filelist = [os.path.join(p, 'kdotpyrc') for p in configpaths if os.path.exists(p)]
	return [f for f in filelist if os.path.isfile(f)]
