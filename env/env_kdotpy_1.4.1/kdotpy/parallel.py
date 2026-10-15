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
from platform import system
import signal
from time import sleep, time as rtime
from datetime import timedelta, datetime as dtime
import multiprocessing as mp
import multiprocessing.dummy as th
from threadpoolctl import threadpool_limits
from typing import Optional, Any, TypeVar, Callable, Iterator, Protocol

from .config import get_all_config, set_all_config
from .cmdargs import sysargv

T = TypeVar('T')  # Generic type

class ValueContainer(Protocol):
	"""Protocol for container type with __len__ and __iter__"""
	def __len__(self) -> int:
		...
	def __iter__(self) -> Iterator:
		...

## Job monitor
job_monitor_enabled: bool = True
def set_job_monitor(state: bool) -> None:
	"""Set global state of 'job monitor' (True or False)"""
	global job_monitor_enabled
	job_monitor_enabled = state
	return

def show_job_monitor(s: str) -> None:
	"""Show 'job monitor' if it is enabled"""
	global job_monitor_enabled
	if job_monitor_enabled:
		sys.stderr.write("***  " + s + "\n")
		sys.stderr.flush()
	return

def job_monitor_k_b(k: Any, b: Any) -> str:
	"""Formatting function for 'job monitor'"""
	if k == 0.0 and b == 0.0:
		return ("%g" % k) if isinstance(k, (float, int)) else str(k)
	elif k == 0.0 and b != 0.0:
		return ("%g" % b) if isinstance(b, (float, int)) else str(b)
	elif k != 0.0 and b == 0.0:
		return ("%g" % k) if isinstance(k, (float, int)) else str(k)
	else:
		k_str = ("%g" % b) if isinstance(b, (float, int)) else str(b)
		b_str = ("%g" % k) if isinstance(k, (float, int)) else str(k)
		return "%s %s" % (k_str, b_str)

def display_etl(s: float | int) -> str:
	"""Format estimated time left (ETL)

	Argument:
	s   Time in seconds
	"""
	s = round(s)
	if s >= 86400:
		d = s // 86400
		h = (s - d * 86400) // 3600
		return "%3id%ih" % (d, h)
	if s >= 3600:
		h = s // 3600
		m = (s - h * 3600) // 60
		return "%2ih%im" % (h, m)
	m = s // 60
	s1 = s - m * 60
	return "%2im%02is" % (m, s1)

def display_etf(s: float | int) -> str:
	"""Format estimated time finished (ETF)

	Argument:
	s   Time in seconds
	"""
	delta = timedelta(seconds = s)
	now = dtime.now()
	etf = now + delta
	today = now.date()
	etfday = etf.date()
	if s <= 600:
		return etf.strftime("%H:%M:%S")
	elif etfday == today:
		return etf.strftime("%H:%M")
	elif s <= 518400:  # 6 * 86400
		return etf.strftime("%a %H:%M")
	else:
		return etf.strftime("%Y-%m-%d")

class Progress:
	"""Container class for progress counter.

	This class shows a progress counter like 2 / 100 and calculates and prints
	the ETL or ETF. The calculation takes into account multiple threads.

	Attributes:
	tm0            Starting time
	tm1            ?
	string         Status string, e.g., 'Doing diagonalization...'
	jobs_done      Integer number of completed jobs (internal)
	n_jobs         Total number of jobs (set on init)
	n_threads      Number of threads. Default is 1.
	update_delay   Float. If nonzero, update counter every this many seconds.
	               If zero, always update if the counter has changed.
	always_update  True or False. If True, also update counter if the number of
	               completed jobs did not increase.
	show_etf       True or False. If True, print ETF. If False, print ETL.
	tcompl         List of length n_threads.
	fancy_counter  True or False. Whether to use a fancy counter (that updates
	               on one location in the terminal). The normal style is to
	               print each update of the counter on a new line. This is set
	               automatically.
	"""
	def __init__(self, string: str, n_jobs: int, n_threads:int = 1, update_delay: float = 0.0, always_update: bool = False, no_output: bool = False):
		global job_monitor_enabled
		self.tm0: float = rtime()
		self.tm1: float = rtime()
		self.string: str = string.strip('\n')
		if not no_output:
			sys.stderr.write(self.string + '...\n')  # write string to sys.stderr
			sys.stderr.flush()
		self.jobs_done: int = 0
		self.n_jobs: int = n_jobs
		self.n_threads: int = n_threads
		self.update_delay: float = update_delay
		self.always_update: bool = always_update
		self.no_output: bool = no_output
		self.show_etf: bool = ("showetf" in sysargv or "monitoretf" in sysargv)
		self.tcompl: list[float | None] = [None for _ in range(0, n_jobs)]
		self.fancy_counter: bool = sys.stderr.isatty() and not job_monitor_enabled

	def show(self, jobsdone: int) -> None:
		"""Show progress and estimated time left/finished (ETL/ETF)

		Argument:
		jobsdone  Integer. Update the counter to this value.
		"""
		if not self.no_output and (self.always_update or jobsdone > self.jobs_done) and (rtime() - self.tm1 > self.update_delay):
			s = "%i / %i" % (jobsdone, self.n_jobs)
			if 1 <= jobsdone <= self.n_jobs:
				est_time_left = self.est_time_left(jobsdone)
				if est_time_left is not None and est_time_left > 0.0:
					if self.show_etf:
						s += " [ETF %s]" % display_etf(est_time_left)
					else:
						s += " [ETL %s]" % display_etl(est_time_left)
			endline = '' if self.fancy_counter and jobsdone < self.n_jobs else '\n'
			clearline = '\r\x1b[K' if self.fancy_counter else ''  # ANSI escape: ESC [ K
			sys.stderr.write(clearline + s + endline)  # write string to sys.stderr
			sys.stderr.flush()
			self.jobs_done = jobsdone
			self.tm1 = rtime()

	def show_do(self, jobsdone: int, value: T) -> T:
		"""Show progress and return the value that has been input without change
		Typical usage: [Progress.show_do(j, f(x)) for j, x in enumerate(xvals)]
		Note that the value is calculated before it is passed to the function;
		hence the +1.
		"""
		self.show(jobsdone + 1)
		return value

	def est_time_left(self, jobsdone: int) -> float | None:
		"""Advanced calculation of ETL, that uses the times at which the previous jobs have completed
		This function takes into account multiple threads."""
		t = rtime()
		if jobsdone > self.jobs_done:
			for j in range(self.jobs_done, jobsdone):
				self.tcompl[j] = t - self.tm0

		thr_time: list[float | None] = [None for thr in range(0, self.n_threads)]
		thr_done: list[int] = [0 for thr in range(0, self.n_threads)]
		thr_total: list[int] = [self.n_jobs // self.n_threads for thr in range(0, self.n_threads)]
		for j in range(0, self.n_jobs % self.n_threads):
			thr_total[j] += 1
		for j in range(0, self.n_jobs):
			if self.tcompl[j] is None:
				break
			thr_time[j % self.n_threads] = self.tcompl[j]
			thr_done[j % self.n_threads] += 1

		if thr_time[0] is None:
			return None
		else:
			thr_est = [thr_time[j] * thr_total[j] / thr_done[j] - (t - self.tm0) for j in range(0, self.n_threads) if thr_time[j] is not None]
			return max(max(thr_est), 0)

### FUNCTIONS FOR PARALLELIZATION ###
# Apply the specified functions over a list of values (vals), with extra
# parameters (f_args) and keyword parameters (f_kwds). We fall back to a simple
# evaluation if the number of processes is set to 1, or if the number of values
# equals 1.

## Signal handling
## This exercise is necessary in order to deal properly with the signal events,
## for example:
##   KeyboardInterrupt   The main process should respond to KeyboardInterrupt,
##                       but the worker process should ignore it. When this
##                       event occurs, the main process should terminate all
##                       worker processes. The program is allowed to continue
##                       with incomplete data. (But usually will fail elsewhere
##                       because of that.)
##   SIGTERM, SIGABRT etc. to the main process
##                       Here, before termination, we raise a custom exception
##                       with the signal number. The main process catches the
##                       exception, terminates to worker processes, and exits
##                       with the appropriate exit code (signal number + 128).
##                       This step is necessary, because if the main process
##                       dies without terminating the worker processes, the
##                       latter can continue working and/or end up as zombies.
##   SIGTERM, etc. to a worker process
##                       If a worker process dies, it emits a SIGCHLD signal
##                       that is caught by the main process. The main process
##                       then terminates the other worker processes and exits.
##                       The signal handler for SIGCHLD has to be reset before
##                       pool.terminate() is called, because terminating the
##                       worker processes this way will also emit SIGCHLD
##                       signals.
## NOTE: On some systems, pool.terminate() will deadlock in some situations.
## This appears to be an (unintended) bug in Python. The bug report
## [https://bugs.python.org/issue29759] might be related.

def init_worker(config: Optional[dict], redefine_signals: bool, job_monitor: Optional[bool]) -> None:
	"""Initializer for worker processes: Set job monitor global state and redefine the signal handler(s)

	Note: To use this function for its intended purpose, it should be given as
	the 'initializer' argument into the multiprocessing.Pool constructor. Its
	'initargs' argument should be set to the tuple (config, redefine_signals,
	job_monitor).

	Arguments:
	config            Dict instance. Set configuration values into the global
	                  configuration. This is needed when the multiprocessing
	                  start method is 'forkserver' or 'spawn'. In that case,
	                  the global configuration would be taken as the empty dict,
	                  which is the value defined in config.py, without
	                  initialization from the config files or command line.
	                  Passing the result from get_all_config() and updating
	                  the global value using set_all_config() copies the correct
	                  values to the workers.
	redefine_signals  True or False. If True, redefine the signals for the
	                  worker process. This is needed if the SignalHandler
	                  context is used.
	job_monitor       True, False, or None. If True or False, set the global
	                  state of the job monitor to this value. If None, ignore.
	"""
	if config is not None:
		set_all_config(config)
	if config is None or config != get_all_config(omit_default=True):
		sys.stderr.write("Warning (init_worker): Configuration was not initialized in parallel worker, or initialization has failed.\n")
	if job_monitor is not None:
		set_job_monitor(job_monitor)
	if redefine_signals:
		signal.signal(signal.SIGTERM, signal.SIG_DFL)
		signal.signal(signal.SIGINT, signal.SIG_IGN)
		if system() != 'Windows':
			signal.signal(signal.SIGCHLD, signal.SIG_IGN)

class TerminateSignal(Exception):
	"""Define a signal handler(s) exception"""
	def __init__(self, signum = None):
		self.signum: int = signum

class SignalHandler:
	"""Context for setting and resetting signal handler(s) in the main process"""
	def sig_handler(self, s, fr):
		raise TerminateSignal(s)

	def __enter__(self):
		if system() == 'Windows':
			self.siglist = [signal.SIGTERM, signal.SIGABRT]
		else:
			self.siglist = [signal.SIGTERM, signal.SIGABRT, signal.SIGUSR1, signal.SIGUSR2, signal.SIGCHLD]
		for s in self.siglist:
			signal.signal(s, self.sig_handler)

	def __exit__(self, exc_type, exc_value, traceback):
		for s in self.siglist:
			signal.signal(s, signal.SIG_DFL)

class NoOpSignalHandler:
	"""Empty context; does not change signal handling"""
	def __enter__(self):
		pass

	def __exit__(self, exc_type, exc_value, traceback):
		pass

def signalstr(s: int) -> str:
	"""String to show when signal s is caught"""
	return "Terminated" if s == signal.SIGTERM else "Interrupted" if s == signal.SIGINT \
		else "Aborted" if s == signal.SIGABRT else "Worker process died" if s == signal.SIGCHLD \
		else "Terminated with signal %i" % s


_long_time = 10000000  # Timeout for pool.apply_async.get()

##
def parallel_apply(
		f: Callable[..., T],
		vals: ValueContainer,
		f_args: tuple | None = None,
		f_kwds: dict | None = None,
		num_processes: int = 1,
		poll_interval: float | int = 1.,
		description: str = "",
		showstatus: bool = True,
		threads: bool = False,
		propagate_interrupt: bool = False) -> list[T]:
	"""Iterative apply a function. Uses either a pool of worker processes or multithreading
	Equivalent to: [f(x, *f_args, **f_kwds) for x in vals]

	Arguments:
	f              Function.
	vals           List of valuess to iterate over.
	f_args         Tuple of extra arguments.
	f_kwds         Dict with extra keyword arguments
	num_processes  Integer. Number of threads.
	poll_interval  Float. Interval in seconds to test how many jobs have been
	               completed.
	description    String. Status message.
	propagate_interrupt  True or False. Whether to propagate (re-raise) a
	                     KeyboardInterrupt event.

	Returns:
	List of function return values.
	"""
	## Default values (f_args and f_kwds)
	if f_args is None:
		f_args = ()
	elif not isinstance(f_args, tuple):
		raise TypeError("Argument f_args must be a tuple or None")
	if f_kwds is None:
		f_kwds = {}
	elif not isinstance(f_kwds, dict):
		raise TypeError("Argument f_kwds must be a dict or None")

	n = len(vals)
	progress = Progress('Calculating' if description == '' else description, n, n_threads = num_processes, no_output= not showstatus)
	all_config = get_all_config(omit_default=True)
	redefine_signals = True
	initargs = (all_config, redefine_signals, job_monitor_enabled)
	if n > 1 and num_processes > 1:
		with SignalHandler(), threadpool_limits(limits=1):
			if threads:
				pool = th.Pool(processes=num_processes)
			else:
				pool = mp.Pool(processes=num_processes, initializer=init_worker, initargs=initargs)
			output = [pool.apply_async(f, args=(x,) + f_args, kwds = f_kwds) for x in vals]

			try:
				while True:
					jobsdone = sum(1 for x in output if x.ready())
					if jobsdone >= n:
						break
					progress.show(jobsdone)
					sleep(poll_interval)
			except TerminateSignal as ex:
				sys.stderr.write("\nERROR (parallel_apply): %s.\n" % signalstr(ex.signum))
				if system() != 'Windows':
					signal.signal(signal.SIGCHLD, signal.SIG_DFL)
				pool.terminate()
				pool.join()
				sleep(1)
				sys.stderr.write("EXIT %i\n" % (128 + ex.signum))
				exit(128 + ex.signum)
			except KeyboardInterrupt:
				sys.stderr.write("\nERROR (parallel_apply): Interrupt.\n")
				if system() != 'Windows':
					signal.signal(signal.SIGCHLD, signal.SIG_DFL)
				progress.always_update = True
				progress.show(jobsdone)
				data = [r.get(_long_time) for r in output if r.ready()]
				pool.terminate()
				pool.join()
				if propagate_interrupt:
					raise
			else:
				if system() != 'Windows':
					signal.signal(signal.SIGCHLD, signal.SIG_DFL)
				progress.show(n)
				data = [r.get() for r in output]
				pool.close()
				pool.join()
	else:
		progress.show(0)
		data = []
		with SignalHandler():
			if system() != 'Windows':
				signal.signal(signal.SIGCHLD, signal.SIG_DFL)
			try:
				with threadpool_limits(limits=1):
					for j, x in enumerate(vals):
						data.append(progress.show_do(j, f(x, *f_args, **f_kwds)))
			except TerminateSignal as ex:
				sys.stderr.write("\nERROR (parallel_apply): %s.\n" % signalstr(ex.signum))
				exit(128 + ex.signum)
			except KeyboardInterrupt:
				sys.stderr.write("\nERROR (parallel_apply): Interrupt.\n")
				if propagate_interrupt:
					raise
			else:
				progress.show(n)
	return data

def dict_plus_array_dict(d_one: dict, d_list: dict[Any, list], j: int) -> dict:
	if d_list == {}:
		return d_one
	out = {}
	for k in d_one:
		out[k] = d_one[k]
	for k in d_list:
		if k in d_one:
			raise KeyError("Duplicate key")
		out[k] = d_list[k][j]
	return out

def parallel_apply_enumerate(
		f: Callable[..., T],
		vals: ValueContainer,
		f_args: tuple | None = None,
		f_kwds: dict | None = None,
		fj_kwds: dict[Any, list] | None = None,
		num_processes: int = 1,
		poll_interval: float | int = 1.,
		description: str = "",
		redefine_signals: bool = True) -> list[T]:
	"""Iteratively apply the function f with indices of the list passed through.
	Equivalent to: [f(j, x, *f_args, **f_kwds) for j, x in enumerate(vals)]

	Arguments:
	f                 Function.
	vals              List of valuess to iterate over.
	f_args            Tuple of extra arguments.
	f_kwds            Dict with extra keyword arguments
	fj_kwds           Dict with lists. For each element, key = fj_kwds[key][j]
	                  is passed to the function f for iteration j.
	num_processes     Integer. Number of threads.
	poll_interval     Float. Interval in seconds to test how many jobs have been
	                  completed.
	description       String. Status message.
	redefine_signals  True or False. If True, redefine signal handling (within
	                  calculation scripts). If False, use standard signal
	                  handling (e.g., with kdotpy batch). Default: True

	Returns:
	List of function return values.
	"""
	## Default values (f_args, f_kwds, and fj_kwds)
	if f_args is None:
		f_args = ()
	elif not isinstance(f_args, tuple):
		raise TypeError("Argument f_args must be a tuple or None")
	if f_kwds is None:
		f_kwds = {}
	elif not isinstance(f_kwds, dict):
		raise TypeError("Argument f_kwds must be a dict or None")
	if fj_kwds is None:
		fj_kwds = {}
	elif not isinstance(fj_kwds, dict):
		raise TypeError("Argument fj_kwds must be a dict or None")

	n = len(vals)
	progress = Progress('Calculating' if description == '' else description, n, n_threads = num_processes)
	handlercontext = SignalHandler if redefine_signals else NoOpSignalHandler
	all_config = get_all_config(omit_default=True)
	initargs = (all_config, redefine_signals, job_monitor_enabled)
	if n > 1 and num_processes > 1:
		with handlercontext(), threadpool_limits(limits=1):
			pool = mp.Pool(processes=num_processes, initializer=init_worker, initargs=initargs)
			output = [pool.apply_async(f, args=(j, x) + f_args, kwds = dict_plus_array_dict(f_kwds, fj_kwds, j)) for j, x in enumerate(vals)]
			try:
				while True:
					jobsdone = sum(1 for x in output if x.ready())
					if jobsdone >= n:
						break
					progress.show(jobsdone)
					sleep(poll_interval)
			except TerminateSignal as ex:
				sys.stderr.write("\nERROR (parallel_apply_enumerate): %s.\n" % signalstr(ex.signum))
				if system() != 'Windows':
					signal.signal(signal.SIGCHLD, signal.SIG_DFL)
				pool.terminate()
				pool.join()
				sleep(0.2)
				exit(128 + ex.signum)
			except KeyboardInterrupt:
				sys.stderr.write("\nERROR (parallel_apply_enumerate): Interrupt.\n")
				if system() != 'Windows':
					signal.signal(signal.SIGCHLD, signal.SIG_DFL)
				progress.always_update = True
				progress.show(jobsdone)
				data = [r.get(_long_time) for r in output if r.ready()]
				pool.terminate()
				pool.join()
			else:
				if system() != 'Windows':
					signal.signal(signal.SIGCHLD, signal.SIG_DFL)
				progress.show(n)
				data = [r.get() for r in output]
				pool.close()
				pool.join()
	else:
		progress.show(0)
		data = []
		with handlercontext():
			if system() != 'Windows':
				signal.signal(signal.SIGCHLD, signal.SIG_DFL)
			try:
				for j, x in enumerate(vals):
					data.append(progress.show_do(j, f(j, x, *f_args, **f_kwds)))
			except TerminateSignal as ex:
				sys.stderr.write("\nERROR (parallel_apply_enumerate): %s.\n" % signalstr(ex.signum))
				exit(128 + ex.signum)
			except KeyboardInterrupt:
				sys.stderr.write("\nERROR (parallel_apply_enumerate): Interrupt.\n")
			else:
				progress.show(n)
	return data

def parallel_apply_expand(
		f: Callable[..., T],
		vals: ValueContainer,
		f_args: list | None = None,
		f_kwds: dict | None = None,
		num_processes: int = 1,
		poll_interval: float | int = 1.,
		description: str = "") -> list[T]:
	"""Apply the function f to a list of tuples, where the tuples are expanded upon	calling the function.
   	Equivalent to: [f(*(x + f_args), **f_kwds) for x vals]

   	Arguments:
	f              Function.
	vals           List of valuess to iterate over.
	f_args         Tuple of extra arguments.
	f_kwds         Dict with extra keyword arguments
	num_processes  Integer. Number of threads.
	poll_interval  Float. Interval in seconds to test how many jobs have been
	               completed.
	description    String. Status message.

	Returns:
	List of function return values.
   	"""
	## Default values (f_args and f_kwds)
	if f_args is None:
		f_args = ()
	elif not isinstance(f_args, tuple):
		raise TypeError("Argument f_args must be a tuple or None")
	if f_kwds is None:
		f_kwds = {}
	elif not isinstance(f_kwds, dict):
		raise TypeError("Argument f_kwds must be a dict or None")

	n = len(vals)
	all_config = get_all_config(omit_default=False)
	redefine_signals = True
	initargs = (all_config, redefine_signals, job_monitor_enabled)
	progress = Progress('Calculating' if description == '' else description, n, n_threads = num_processes)
	for x in vals:
		if not isinstance(x, tuple):
			raise ValueError("Elements of vals must be tuples")
	if n > 1 and num_processes > 1:
		with SignalHandler(), threadpool_limits(limits=1):
			pool = mp.Pool(processes=num_processes, initializer=init_worker, initargs=initargs)
			output = [pool.apply_async(f, args=x + f_args, kwds = f_kwds) for x in vals]

			try:
				while True:
					jobsdone = sum(1 for x in output if x.ready())
					if jobsdone >= n:
						break
					progress.show(jobsdone)
					sleep(poll_interval)
			except TerminateSignal as ex:
				sys.stderr.write("\nERROR (parallel_apply_expand): %s.\n" % signalstr(ex.signum))
				if system() != 'Windows':
					signal.signal(signal.SIGCHLD, signal.SIG_DFL)
				pool.terminate()
				pool.join()
				sleep(0.2)
				exit(128 + ex.signum)
			except KeyboardInterrupt:
				sys.stderr.write("\nERROR (parallel_apply_expand): Interrupt.\n")
				if system() != 'Windows':
					signal.signal(signal.SIGCHLD, signal.SIG_DFL)
				progress.always_update = True
				progress.show(jobsdone)
				data = [r.get(_long_time) for r in output if r.ready()]
				pool.terminate()
				pool.join()
			else:
				if system() != 'Windows':
					signal.signal(signal.SIGCHLD, signal.SIG_DFL)
				progress.show(n)
				data = [r.get() for r in output]
				pool.close()
				pool.join()
	else:
		progress.show(0)
		data = []
		with SignalHandler():
			if system() != 'Windows':
				signal.signal(signal.SIGCHLD, signal.SIG_DFL)
			try:
				for j, x in enumerate(vals):
					data.append(progress.show_do(j, f(*(x + f_args), **f_kwds)))
			except TerminateSignal as ex:
				sys.stderr.write("\nERROR (parallel_apply_expand): %s.\n" % signalstr(ex.signum))
				exit(128 + ex.signum)
			except KeyboardInterrupt:
				sys.stderr.write("\nERROR (parallel_apply_expand): Interrupt.\n")
			else:
				progress.show(n)
	return data
