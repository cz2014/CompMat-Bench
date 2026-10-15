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
from time import sleep, perf_counter, time as rtime
from multiprocessing.pool import Pool as ProcessPool, ThreadPool
from queue import PriorityQueue
from platform import system
import signal
from threadpoolctl import threadpool_limits
from typing import Callable, TypeVar, Generic

from .config import get_config_int, get_all_config
from .parallel import show_job_monitor, TerminateSignal, signalstr, init_worker, job_monitor_enabled
from . import cmdargs

T = TypeVar('T')

class TaskWrapper(Generic[T]):
    """Wraps around any tasks work load function, taking care of any arguments.
    This object is can be pickled and sent to other processes."""
    def __init__(self, func: Callable[..., T], *args, **kwds):
        self.func: Callable[..., T] = func
        self.args: tuple = args
        self.kwds: dict = kwds

    def run(self) -> T:
        """Run the task (function with arguments). Typically executed
        within the scope of a worker thread/process."""
        return self.func(*self.args, **self.kwds)


class Task:
    """Class that holds function handles and parallelization settings (from Model class)
     and parameters (from DiagDataPoint).
     Takes care of status information output and error handling."""
    def __init__(
            self,
            queue: 'TaskManager',
            name: str | None = None,
            worker_func: Callable | None = None,
            callback_func: Callable | None = None,
            worker_type: str | None = None,
            n_proc: int = 1,
            n_threads: int = 1,
            gpu_workers: int = 0,
            priority: int | tuple[int, ...] = 0) -> None:
        """Create and enqueue a new Task."""
        if not isinstance(queue, TaskManager):
            raise TypeError('Task creation: Arg queue must specify a queue of class TaskManager')
        self.queue: 'TaskManager' = queue
        self.name: str | None = name if name is not None else worker_func.__name__ + '_%d' % (perf_counter() * 1e9)
        if not callable(worker_func):
            raise TypeError('Task: Worker function must be a callable function handle.')
        self.worker_func: Callable = worker_func  # must return a single value
        self.callback: Callable = self.wrap_callback(callback_func)  # must return a new Task or None and save the result from worker_func into an object
        self.worker_type: str | None = worker_type
        self.n_proc: int = n_proc
        self.n_threads: int = n_threads
        self.gpu_workers: int = gpu_workers
        self.priority: int | tuple[int, ...] = priority
        self.start_time: float | None = None
        self.error_counter: int = 0
        self.retries: int = get_config_int('task_retries')
        queue.put(self)

    def __str__(self) -> str:
        """Print a human readable label for this task. Note that start time is not a real-time stamp."""
        if self.start_time is None:
            return 'Task: %s (%s)' % (self.name, self.priority)
        else:
            return 'Task: %s (%s), started: %f' % (self.name, self.priority, self.start_time)

    def __lt__(self, other: 'Task') -> bool:
        """Definition of priority ordering.
        In case priority (numeric or tuple) is equal,
        do tasks with use less threads first."""
        if self.priority == other.priority:
            return self.threadsum < other.threadsum
        return self.priority < other.priority

    def __eq__(self, other: 'Task') -> bool:
        return self.priority == other.priority and self.threadsum == other.threadsum

    @property
    def threadsum(self) -> int:
        """Total threads used by this task (processes times threads per process)."""
        return self.n_threads * self.n_proc

    def run(self) -> int:
        """Run this task on a suitable parallel worker pool or sequentially in the main process.
        Optional status output."""
        self.start_time = rtime()
        p_pool = self.queue.p_pool
        t_pool = self.queue.t_pool
        show_job_monitor("Entering %s..." % self.name)
        if self.worker_type == 'process' and isinstance(p_pool, ProcessPool):
            p_pool.apply_async(self.worker_func, callback=self.callback, error_callback=self.error_callback)
            return self.threadsum
        elif self.worker_type == 'thread' and isinstance(t_pool, ThreadPool):
            t_pool.apply_async(self.worker_func, callback=self.callback, error_callback=self.error_callback)
            return self.n_threads
        else:
            # work on this task in sequential mode
            self.callback(self.worker_func())
            return self.threadsum

    def wrap_callback(self, org_cb: Callable) -> Callable:
        """Modify callback function handle."""
        def extended_callback(*args, **kwds):
            """Add task finish console output and notify queue about finished task."""
            retval = org_cb(*args, **kwds)  # execute original callback
            show_job_monitor("Finished %s (%.3f s)" % (self.name, rtime() - self.start_time))
            self.queue.done(self)
            return retval  # return original callback result
        return extended_callback

    def error_callback(self, exception: BaseException) -> None:
        """Alternative callback to catch errors during task execution.
        Notify about failed tasks and retry for configured number of tries.
        Finally, skip task."""
        self.error_counter += 1
        if self.error_counter > self.retries:
            show_job_monitor(
                "EXCEPTION in %s! (%.3f s)[%s] Skipped after %d tries." %
                (self.name, rtime() - self.start_time, exception, self.error_counter)
            )
            self.queue.done(self, skipped=True)
        else:
            show_job_monitor("EXCEPTION in %s! (%.3f s)[%s] Restarting..." % (self.name, rtime() - self.start_time, exception))
            self.run()


class TaskManager(PriorityQueue):
    """Extended Queue class to schedule Task objects and handle worker pools.

    Only use this class in a main process/thread, as it creates worker pools,
    which is not allowed from within an existing worker pool.

    Redefinition of the base class changes scheduling
    behaviour between LIFO, FIFO and Priority.
    """
    def __init__(self, max_total_threads: int | None = None, handle_sigchld: bool = True):
        """Create extended Queue class to schedule Task objects and handle worker pools.

        See documentation for TaskManager class for more information.

        Arguments:
        max_total_threads  Integer.
        handle_sigchld     True (default) or False. The value tells the
                           TaskManager to redefine SIGCHLD to terminate on that
                           signal. This is needed to make multiprocessing handle
                           the case that a child process dies in a graceful way.
                           However, this can interfere with some (external)
                           solvers, like jax. For those solvers, handle_sigchld
                           should be set to False. The value does not affect
                           behaviour on Windows.
        """
        super().__init__()
        if max_total_threads is None:
            # TODO: Parse these command-line arguments earlier and pass hem as arguments to __init__()
            n_threads = cmdargs.threads()
            n_cpus, max_cpus = cmdargs.cpus()
            n_gpus = cmdargs.gpu_workers()
            self.n_threads: int = n_threads if n_threads is not None else 1
            self.max_workers: int = n_cpus
            self.max_gpu_workers: int = n_gpus if n_gpus is not None else n_cpus
            self.max_total_threads: int = n_cpus * (n_threads if n_threads is not None else 1)
        else:
            self.n_threads: int = 1
            self.max_workers: int = max_total_threads
            self.max_total_threads: int = max_total_threads
            self.max_gpu_workers: int = max_total_threads
        self.running_threads: int = 0
        self.running_workers: int = 0
        self.running_gpu_workers: int = 0
        self.skip_counter: int = 0
        self.p_pool: ProcessPool | None = None
        self.t_pool: ThreadPool | None = None
        self.handle_sigchld: bool = handle_sigchld

    def sig_handler(self, s, fr):
        raise TerminateSignal(s)

    def __enter__(self):
        """Start up pools when entering task manager context."""
        if system() == 'Windows':
            self.siglist = [signal.SIGTERM, signal.SIGABRT]
        elif self.handle_sigchld:
            self.siglist = [signal.SIGTERM, signal.SIGABRT, signal.SIGUSR1, signal.SIGUSR2, signal.SIGCHLD]
        else:
            self.siglist = [signal.SIGTERM, signal.SIGABRT, signal.SIGUSR1, signal.SIGUSR2]
        for s in self.siglist:
            signal.signal(s, self.sig_handler)
        all_config = get_all_config(omit_default=True)
        redefine_signals = True
        initargs = (all_config, redefine_signals, job_monitor_enabled)
        if self.max_workers > 1:
            show_job_monitor("Starting worker pools...")
            with threadpool_limits(limits=self.n_threads):
                self.p_pool = ProcessPool(self.max_workers, initializer=init_worker, initargs=initargs)
                self.t_pool = ThreadPool(self.max_workers)
        self.start_time = rtime()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close worker pools when leaving task manager context.
        Print run time information and skipped jobs."""
        show_job_monitor("Total compute time %d s." % (rtime() - self.start_time))
        if self.skip_counter > 0:
            show_job_monitor("Skipped %d tasks." % self.skip_counter)
        for s in self.siglist:
            signal.signal(s, signal.SIG_DFL)
        if system() != 'Windows':
            signal.signal(signal.SIGCHLD, signal.SIG_DFL)
        if self.max_workers > 1:
            if exc_type is None:  # Pool exits normally
                self.p_pool.close()
                self.t_pool.close()
            else:  # Covers also: isinstance(exc_val, (KeyboardInterrupt, Exception))
                sys.stderr.write("TaskManager: %s\nTerminating pools...\n" % exc_type.__name__)
                self.p_pool.terminate()
                self.t_pool.terminate()
            self.p_pool.join()
            self.t_pool.join()

            if isinstance(exc_val, TerminateSignal):
                sys.stderr.write("EXIT %i (%s)\n" % (128 + exc_val.signum, signalstr(exc_val.signum)))
                exit(128 + exc_val.signum)
            elif isinstance(exc_val, KeyboardInterrupt):
                sys.stderr.write("EXIT %i (%s)\n" % (128 + signal.SIGINT, signalstr(signal.SIGINT)))
                exit(128 + signal.SIGINT)
            # In all other cases unhandled errors are propagated automatically

    def done(self, task: Task, skipped: bool = False) -> None:
        """Keep track of finished jobs and available resources."""
        super().task_done()
        self.running_threads -= task.threadsum
        self.running_workers -= task.n_proc
        self.running_gpu_workers -= task.gpu_workers
        if skipped:
            self.skip_counter += 1

    def do_all(self) -> None:
        """Run all tasks in queue, blocking the calling thread until none are left.

        This blocks the calling thread until there are no tasks left in the
        queue. Tasks with process and thread worker strategy are executed
        outside of the calling thread, but tasks without a parallel worker
        strategy are executed in the calling thread."""
        while self.unfinished_tasks:
            if not self.empty() and self.running_workers < self.max_workers:
                task = None
                if self.running_gpu_workers >= self.max_gpu_workers:
                    # GPU is fully in use, find CPU only tasks.
                    gpu_task_list = []
                    while not self.empty():
                        task = self.get()
                        if task.gpu_workers > 0:
                            gpu_task_list.append(task)
                            task = None
                        else:
                            break  # Task found, stop searching through Queue
                    for gtask in gpu_task_list:
                        self.put(gtask)  # requeue removed GPU tasks
                        super().task_done()  # requeueing increases unfinished tasks counter.
                else:
                    task = self.get()

                # Check free CPU and GPU resources:
                if task is not None:
                    if self.running_threads + task.threadsum <= self.max_total_threads \
                            and self.running_gpu_workers + task.gpu_workers <= self.max_gpu_workers:
                        with threadpool_limits(limits=self.n_threads):
                            task.run()  # run
                        self.running_threads += task.threadsum
                        self.running_workers += 1
                        self.running_gpu_workers += task.gpu_workers
                    else:
                        self.put(task)  # this task was to big to fit right now, so requeue it
                        super().task_done()  # as requeueing increases the unfinished task counter, reduce it
            else:
                # All workers busy. Nothing to be enqueued.
                # Send this thread to sleep to yield cpu time.
                # Do not sleep too long, as this would slow down queueing,
                # especially with fast tasks and few workers.
                sleep(0.005)
