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
import os.path
from platform import system
import subprocess
from time import time as rtime
import re
import shlex

from . import run

testdir = 'test'
_kdotpy_cmd = 'kdotpy'
_verbose = False
_showcmd = False

def run_test(cmd: str, args: list[str], append_outdir: bool = True) -> subprocess.CompletedProcess:
	if append_outdir:
		args = args + ['outdir', testdir]
	if _verbose:
		args = args + ['verbose']
	if not sys.executable:
		raise OSError("Python executable (sys.executable) is undefined")
	args = [sys.executable, '-m', _kdotpy_cmd, cmd] + args
	try:
		cp = subprocess.run(args)
	except OSError:
		if system() != 'Windows':
			args = " ".join([shlex.quote(arg) for arg in args])
		cp = subprocess.run(args, shell=True)
	return cp

def cmd_to_string(cmd: str, args: list[str], append_outdir: bool = True):
	if append_outdir:
		args = args + ['outdir', testdir]
	if _verbose:
		args = args + ['verbose']
	cmd_str = _kdotpy_cmd + " " + cmd
	return (cmd_str + " " + " ".join([shlex.quote(arg) for arg in args]))

def with_default_config(args: list[str]) -> list[str]:
	"""Insert 'config default' into list of command line arguments before any other config argument"""
	i = args.index('config') if 'config' in args else len(args)
	if i + 1 < len(args) and args[i + 1] == 'default':
		return args
	if '--' in args:  # do not insert beyond --
		i = min(i, args.index('--'))
	return args[:i] + ['config', 'default'] + args[i:]


## Class definitions compatible with pytest
class TestRuns:
	def test_2d_qw(self, get_cmd: bool = False) -> str | None:
		cmd = "2d"
		args = "8o obs subbande1e2h1h2l1 overlaps erange -100 20 k 0 0.5 / 25 kphi 45 zres 0.25 targetenergy -35 " \
		       "neig 20 split 0.01 llayer 5 11 5 mlayer HgCdTe 68% HgMnTe 2.4% HgCdTe 68% msubst CdZnTe 4% noax " \
		       "out .test_2d_qw dos localdos dostemp 2 char legend extrema symmetrize bhz " \
		       "config fig_colorbar_method=file;dos_quantity=e;dos_unit=cm".split(" ")
		args = with_default_config(args)
		if get_cmd:
			return cmd_to_string(cmd, args)
		cp = run_test(cmd, args)
		exitcode = cp.returncode
		assert exitcode == 0

	def test_2d_qw_bia(self, get_cmd: bool = False) -> str | None:
		cmd = "2d"
		args = "8o obs subbande1e2h1h2l1 overlaps erange -100 20 k -0.5 0.5 / 50 kphi 45 zres 0.25 targetenergy -35 " \
		       "neig 20 split 0.01 llayer 5 11 5 mlayer HgCdTe 68% HgMnTe 2.4% HgCdTe 68% msubst CdZnTe 4% noax " \
		       "out .test_2d_qw_bia char legend bia config fig_colorbar_method=file".split(" ")
		args = with_default_config(args)
		if get_cmd:
			return cmd_to_string(cmd, args)
		cp = run_test(cmd, args)
		exitcode = cp.returncode
		assert exitcode == 0

	def test_2d_qw_2(self, get_cmd: bool = False) -> str | None:
		cmd = "2d"
		args = "8o obs subbande1e2h1h2l1 overlaps erange -100 20 k -0.48 0.48 / 24 kphi 45 zres 0.25 targetenergy -35 " \
		       "neig 20 split 0.01 llayer 5 11 5 mlayer HgCdTe 68% HgMnTe 2.4% HgCdTe 68% msubst CdZnTe 4% noax " \
		       "out .test_2d_qw_2 char legend config fig_colorbar_method=file".split(" ")
		args = with_default_config(args)
		if get_cmd:
			return cmd_to_string(cmd, args)
		cp = run_test(cmd, args)
		exitcode = cp.returncode
		assert exitcode == 0

	def test_run(self, get_cmd: bool = False) -> str | None:
		cmd = "2d"
		args = "8o obs subbande1e2h1h2l1 overlaps erange -100 20 k 0 0.5 / 25 kphi 45 zres 0.25 targetenergy -35 " \
		       "neig 20 split 0.01 llayer 5 11 5 mlayer HgCdTe 68% HgMnTe 2.4% HgCdTe 68% msubst CdZnTe 4% noax " \
		       "out .test_run dos dostemp 2 char legend symmetrize " \
		       "config dos_quantity=e;dos_unit=cm".split(" ")
		args = with_default_config(args)
		args = args + ['outdir', testdir]
		if _verbose:
			args = args + ['verbose']
		if get_cmd:
			return f'kdotpy.run("{cmd}", ' + ', '.join([f'"{arg}"' for arg in args]) + ")  # in Python"
		exitcode = run(cmd, *args)
		assert exitcode == 0

	def test_2d_polar(self, get_cmd: bool = False) -> str | None:
		cmd = "2d"
		args = "8o obs orbitalrgb erange -100 20 k 0 0.5 / 25 kphi 0 90 / 6 zres 0.25 targetenergy -35 neig 20 " \
		       "split 0.01 llayer 5 11 5 mlayer HgCdTe 68% HgMnTe 2.4% HgCdTe 68% msubst CdZnTe 4% noax " \
		       "out .test_2d_polar char legend plotstyle spinxy extrema".split(" ")
		args = with_default_config(args)
		if get_cmd:
			return cmd_to_string(cmd, args)
		cp = run_test(cmd, args)
		exitcode = cp.returncode
		assert exitcode == 0

	def test_2d_orient(self, get_cmd: bool = False) -> str | None:
		cmd = "2d"
		args = "8o obs orbitalrgb erange -100 50 k 0 0.5 / 25 kphi 0 360 / 24 zres 0.25 targetenergy -35 neig 20 " \
		       "split 0.01 llayer 5 8 5 mlayer HgCdTe 68% HgTe HgCdTe 68% msubst CdZnTe 4% noax " \
		       "out .test_2d_orient char legend orient 30d 111 symmetrytest".split(" ")
		args = with_default_config(args)
		if get_cmd:
			return cmd_to_string(cmd, args)
		cp = run_test(cmd, args)
		exitcode = cp.returncode
		assert exitcode == 0

	def test_2d_cartesian(self, get_cmd: bool = False) -> str | None:
		cmd = "2d"
		args = "8o obs jz erange -100 20 kx 0 0.5 / 20 ky 0 0.5 / 20 zres 0.25 targetenergy -35 neig 24 split 0.01 " \
		       "llayer 5 11 5 mlayer HgCdTe 68% HgMnTe 2.4% HgCdTe 68% msubst CdZnTe 4% noax " \
		       "out .test_2d_cartesian char legend extrema plotwf separate zero".split(" ")
		args = with_default_config(args)
		if get_cmd:
			return cmd_to_string(cmd, args)
		cp = run_test(cmd, args)
		exitcode = cp.returncode
		assert exitcode == 0

	def test_2d_offset(self, get_cmd: bool = False) -> str | None:
		cmd = "2d"
		args = "8o obs jz erange -100 20 kx 0.1 ky 0 0.5 / 20 zres 0.25 targetenergy -35 neig 24 split 0.01 " \
		       "llayer 5 11 5 mlayer HgCdTe 68% HgMnTe 2.4% HgCdTe 68% msubst CdZnTe 4% noax " \
		       "out .test_2d_offset char legend extrema dos plotwf separate zero".split(" ")
		args = with_default_config(args)
		if get_cmd:
			return cmd_to_string(cmd, args)
		cp = run_test(cmd, args)
		exitcode = cp.returncode
		assert exitcode == 0

	def test_ll_legacy(self, get_cmd: bool = False) -> str | None:
		cmd = "ll"
		args = "8o temp 0.1 zres 0.25 llayer 10 12 10 b 0 1.6 // 16 k 0 mlayer HgCdTe 68% HgTe HgCdTe 68% " \
		       "erange -90 50 targetenergy 10 neig 120 nll 10 msubst CdTe legend char split 0.01 obs llindex " \
		       "out .test_ll_legacy dos cardens 0.002 localdos broadening 0.5 20% lllegacy " \
		       "config plot_transitions_labels=false;fig_ticks_major=more;fig_ticks_minor=normal;fig_unit_format=();plot_dos_units_negexp=true".split(" ")
		args = with_default_config(args)
		if get_cmd:
			return cmd_to_string(cmd, args)
		cp = run_test(cmd, args)
		exitcode = cp.returncode
		assert exitcode == 0

	def test_ll_axial(self, get_cmd: bool = False) -> str | None:
		cmd = "ll"
		args = "8o temp 0.1 zres 0.25 llayer 10 12 10 b 0 1.6 // 16 k 0 mlayer HgCdTe 68% HgTe HgCdTe 68% " \
		       "erange -90 50 targetenergy 10 neig 120 nll 10 msubst CdTe legend char split 0.01 obs llindex berry " \
		       "out .test_ll_axial transitions dos cardens 0.002 localdos broadening 0.5 20% " \
		       "config plot_transitions_labels=false;fig_ticks_major=more;fig_ticks_minor=normal;fig_unit_format=();plot_dos_units_negexp=true".split(" ")
		args = with_default_config(args)
		if get_cmd:
			return cmd_to_string(cmd, args)
		cp = run_test(cmd, args)
		exitcode = cp.returncode
		assert exitcode == 0

	def test_ll_bia(self, get_cmd: bool = False) -> str | None:
		cmd = "ll"
		args = "8o temp 0.1 zres 0.25 llayer 10 12 10 b 0 1.6 // 16 k 0 mlayer HgCdTe 68% HgTe HgCdTe 68% " \
		       "erange -90 50 targetenergy 10 neig 50 nll 5 bia msubst CdTe legend char split 0.01 obs llavg berry " \
		       "out .test_ll_bia transitions dos cardens 0.002 localdos broadening 0.5 20% " \
		       "config plot_transitions_labels=false;fig_ticks_major=more;fig_ticks_minor=normal;fig_unit_format=();plot_dos_units_negexp=true".split(" ")
		args = with_default_config(args)
		if get_cmd:
			return cmd_to_string(cmd, args)
		cp = run_test(cmd, args)
		exitcode = cp.returncode
		assert exitcode == 0

	def test_bulk_ll(self, get_cmd: bool = False) -> str | None:
		cmd = "bulk-ll"
		args = "8o b 0 10 / 100 split 0.01 mater HgTe msubst CdZnTe 4% neig 30 nll 10 " \
		       "out .test_bulk_ll erange -20 20 obs bindex legend".split(" ")
		args = with_default_config(args)
		if get_cmd:
			return cmd_to_string(cmd, args)
		cp = run_test(cmd, args)
		exitcode = cp.returncode
		assert exitcode == 0

	def test_bulk(self, get_cmd: bool = False) -> str | None:
		cmd = "bulk"
		args = "8o overlaps obs jz erange -100 20 k 0 0.5 / 50 kphi 45 split 0.01 mater HgTe msubst CdZnTe 4% noax " \
		       "out .test_bulk char legend symmetrize".split(" ")
		args = with_default_config(args)
		if get_cmd:
			return cmd_to_string(cmd, args)
		cp = run_test(cmd, args)
		exitcode = cp.returncode
		assert exitcode == 0

	def test_bulk_3d(self, get_cmd: bool = False) -> str | None:
		cmd = "bulk"
		args = "8o obs jz erange -100 20 kx 0 0.15 / 15 ky 0 0.15 / 15 kz 0 0.15 / 15 split 0.01 mater HgTe strain -0.3% noax " \
		       "out .test_bulk_3d char legend extrema dos".split(" ")
		args = with_default_config(args)
		if get_cmd:
			return cmd_to_string(cmd, args)
		cp = run_test(cmd, args)
		exitcode = cp.returncode
		assert exitcode == 0

	def test_merge(self, get_cmd: bool = False) -> str | None:
		cmd = "merge"
		datadir = testdir if os.path.isdir(testdir) else '.'
		args = ["out", ".test_merge", "reconnect", "-25", "2", 'outdir', testdir, "--",
		        os.path.join(datadir, "output.test_2d_qw.xml"), os.path.join(datadir, "output.test_2d_qw_2.xml")]
		# TODO: Run test_2d_qw and test_2d_qw_2 when data files are not found
		args = with_default_config(args)
		if get_cmd:
			return cmd_to_string(cmd, args, append_outdir=False)
		cp = run_test(cmd, args, append_outdir=False)
		exitcode = cp.returncode
		assert exitcode == 0

	def test_compare_2d(self, get_cmd: bool = False) -> str | None:
		cmd = "compare"
		datadir = testdir if os.path.isdir(testdir) else '.'
		args = ["out", ".test_compare_2d", "legend", 'outdir', testdir, "--",
		        os.path.join(datadir, "output.test_2d_qw.xml"), "vs",
		        os.path.join(datadir, "output.test_2d_qw_2.xml")]
		# TODO: Run test_2d_qw and test_2d_qw_2 when data files are not found
		args = with_default_config(args)
		if get_cmd:
			return cmd_to_string(cmd, args, append_outdir=False)
		cp = run_test(cmd, args, append_outdir=False)
		exitcode = cp.returncode
		assert exitcode == 0

	def test_compare_ll(self, get_cmd: bool = False) -> str | None:
		cmd = "compare"
		datadir = testdir if os.path.isdir(testdir) else '.'
		args = ["out", ".test_compare_ll", "legend", "erange", "-90", "-50", 'outdir', testdir, "--",
		        os.path.join(datadir, "output.test_ll_axial.xml"), "vs",
		        os.path.join(datadir, "output.test_ll_bia.xml")]
		# TODO: Run test_ll_axial and test_ll_bia when data files are not found
		args = with_default_config(args)
		if get_cmd:
			return cmd_to_string(cmd, args, append_outdir=False)
		cp = run_test(cmd, args, append_outdir=False)
		exitcode = cp.returncode
		assert exitcode == 0

	def test_2d_selfcon(self, get_cmd: bool = False) -> str | None:
		cmd = "2d"
		args = "8o obs orbitalrgb erange -100 60 k 0 0.5 / 50 kphi 45 zres 0.25 targetenergy -35 neig 20 " \
		       "split 0.01 llayer 5 11 5 mlayer HgCdTe 68% HgMnTe 2.4% HgCdTe 68% msubst CdZnTe 4% noax " \
		       "out .test_2d_selfcon dos dostemp 2 char legend selfcon cardens 0.002 plotqz " \
		       "config selfcon_full_diag=false".split(" ")
		args = with_default_config(args)
		if get_cmd:
			return cmd_to_string(cmd, args)
		cp = run_test(cmd, args)
		exitcode = cp.returncode
		assert exitcode == 0

	def test_batch(self, get_cmd: bool = False) -> str | None:
		cmd1 = "batch"
		cmd2 = "2d"
		args1 = "@d 7 9 / 2".split(" ")
		args2 = "8o obs jz erange -100 20 kx 0 0.5 / 20 zres 0.25 targetenergy -35 neig 24 split 0.01 " \
		        "llayer 5 @d 5 mlayer HgCdTe 68% HgTe HgCdTe 68% msubst CdZnTe 4% noax " \
		        "out .test_batch.@0of@@ char legend".split(" ")
		args = args1 + ['do', _kdotpy_cmd, cmd2] + args2
		args = with_default_config(args)
		if get_cmd:
			return cmd_to_string(cmd1, args)
		cp = run_test(cmd1, args)
		exitcode = cp.returncode
		assert exitcode == 0

	def test_1d(self, get_cmd: bool = False) -> str | None:
		cmd = "1d"
		args = "8o axial mlayer HgCdTe 68% HgTe HgCdTe 68% msubst CdZnTe 4% llayer 5 7 5 zres 0.25 w 10 wres 0.5 " \
		       "k 0 0.3 / 6 symmetrize split 0.01 targetenergy -30 neig 30 obs sz out .test_1d bia legend".split(" ")
		args = with_default_config(args)
		if get_cmd:
			return cmd_to_string(cmd, args)
		cp = run_test(cmd, args)
		exitcode = cp.returncode
		assert exitcode == 0


## If called from the command line
def main(args: list[str] | None = None) -> None:
	global _showcmd, _verbose
	sysargv = sys.argv[2:] if args is None else args
	_showcmd = False  # Initialize at run-time
	_verbose = False  # Initialize at run-time

	# Terminal colors
	COLOR_DISPLAY = sys.stdout.isatty()
	cred = '\x1b[1;31m' if COLOR_DISPLAY else ''
	cgreen = '\x1b[1;32m' if COLOR_DISPLAY else ''
	cyellow = '\x1b[1;33m' if COLOR_DISPLAY else ''
	cblue = '\x1b[1;34m' if COLOR_DISPLAY else ''
	cpurple = '\x1b[1;35m' if COLOR_DISPLAY else ''
	ccyan = '\x1b[1;36m' if COLOR_DISPLAY else ''
	cwhite = '\x1b[1;37m' if COLOR_DISPLAY else ''
	creset = '\x1b[0m' if COLOR_DISPLAY else ''

	tr = TestRuns()
	all_tests = [member[5:] for member in dir(tr) if member.startswith('test')]
	# Change default alphabetical test sort order:
	# Move tests that depend on input files from other runs to the end.
	# Test run order can be changed by order in sysargv
	independent_tests = []
	dependent_tests = []
	for test in all_tests:
		if 'compare' in test or 'merge' in test:
			dependent_tests.append(test)
		else:
			independent_tests.append(test)
	all_tests = independent_tests + dependent_tests
	del independent_tests, dependent_tests

	invalid_testid = False
	if len(sysargv) >= 1:
		do_tests = []
		if sysargv[0].lower() == 'list':
			print("Valid test ids: %s\n" % (", ".join(all_tests)))
			exit(0)
		for a in sysargv:
			a = a.replace('-', '_')  # Also allow more comfortable input names with '-' instead of '_'.
			if a in all_tests:
				do_tests.append(a)
			elif a == 'verbose':
				_verbose = True
			elif 'python' in a:
				m = re.search(r"python(3([.][0-9]+)?([.][0-9]+)?)?$", a)
				if m is None:
					sys.stderr.write("ERROR (kdotpy-test.py): '%s' is not a valid Python command. Allowed commands: 'python', 'python3', 'python3.x', 'python3.x.y' (where x, y are numbers).\n" % a)
				else:
					_python_cmd = a
			elif a.startswith('showcmd'):
				_showcmd = True
			else:
				sys.stderr.write("ERROR (kdotpy-test.py): Test id '%s' is not valid.\n" % a)
				invalid_testid = True
		if len(do_tests) == 0:
			if not invalid_testid:
				# catch the case that verbose or showcmd option was given without any test id (i.e. run all tests).
				do_tests = all_tests
			else:
				sys.stderr.write("ERROR (kdotpy-test.py): No valid test ids given.\nValid ids: %s\n" % (", ".join(all_tests)))
				exit(2)
	else:
		do_tests = all_tests
	print(("# " if _showcmd else "") + "------ kdotpy test suite ------")
	runtimes = []
	stati = []
	t0 = rtime()
	for testid in do_tests:
		testfunc = getattr(tr, 'test_' + testid)
		if _showcmd:
			print("# Test %s:" % testid)
			print(testfunc(get_cmd = True))
			print()
			continue
		t1 = rtime()
		try:
			print("Starting test %s..." % testid)
			testfunc()
		except AssertionError:
			print("%sTest %s%s: %sFailed%s" % (cwhite, cpurple, testid, cred, creset), end = '\n\n')
			stati.append(False)
			runtimes.append(rtime() - t1)
		except:
			raise
		else:
			print("%sTest %s%s: %sSuccess%s" % (cwhite, cpurple, testid, cgreen, creset), end = '\n\n')
			stati.append(True)
			runtimes.append(rtime() - t1)

	if _showcmd:
		exit(0)

	if len(runtimes) > 0:
		# summarize test timings:
		# column widths:
		cw = [max(10, len(max(do_tests, key=len)) + 1), 9, 13]
		print('Test'.center(cw[0]) + '|' + 'Status'.center(cw[1]) + '|' + 'Runtime (s)'.center(cw[2]))
		print(('-' * cw[0]) + '+' + ('-' * cw[1]) + '+' + ('-' * cw[2]))
		for i, (testid, status, runtime) in enumerate(zip(do_tests, stati, runtimes)):
			color = cblue if i % 2 else ccyan
			print(color + testid.ljust(cw[0]) + creset, end = '|')
			if status:
				print(cgreen + 'success'.center(cw[1]) + creset, end = '|')
			else:
				print(cred + 'failed'.center(cw[1]) + creset, end = '|')
			print(color + ('%.1f' % runtime).rjust(cw[2]) + creset)
		print('-' * (sum(cw) + len(cw) - 1))
		print("Total run time: %ds" % (rtime() - t0))

	if invalid_testid:
		sys.stderr.write("ERROR (kdotpy-test.py): Argument list contains invalid test id.\nValid ids: %s\n" % (", ".join(all_tests)))

if __name__ == '__main__':
	main()

