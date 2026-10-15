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
import numpy as np
from typing import Any, Callable

from . import resumetools as resume
from .types import PhysParams, MatrixT, DiagSolver
from .cmdargs import sysargv, resume_from
from .config import get_config, get_config_num, get_config_bool
from .vector import Vector, VectorGrid, locations_index
from .diagonalization import DiagDataPoint
from .symbolic import SymbolicHamiltonian
from .parallel import job_monitor_k_b, Progress
from . import hamiltonian as hm
from .diagonalization.diagsolver import CupyShiftInvEigshSolver, JaxEighSolver, SLEPCGPUSolver
from .lltools import delta_n_ll, scaleup_eivec, scaleup_full_eivec
from .transitions import get_transitions, get_transitions_full
from .berry import berrycurv_k, chernnumber_ll, chernnumber_ll_full
from .currents import set_disp_derivatives_dhdk, set_disp_derivatives_dhdk_obs

from .tasks import Task, TaskWrapper, TaskManager

class ModelBase:
    """Basic dummy class for calculation models. This contains all function definitions independent of the exact model (e.g. 1D dispersion, LL mode, etc).

    Developer note:
    This class works together with classes DiagData(Point) and tasks.py to
    enable flexible configuration of multiprocessing and multithreading in
    kdotpy. While a DiagDataPoint instance holds the parameters for the model
    that are unique to that point, the Model class defines the which steps need
    to be calculated for all points (e.g. Hamiltonian construction,
    diagonalization, postprocessing). Tasks hold the information needed to
    dispatch all model steps for each DiagDataPoint to worker threads/processes.

    Since sending objects between processes in python involves (un-)pickling all
    the necessary information into a byte stream, some wrappers are needed to
    correctly execute the code in worker and main threads. While the actual
    calculation workload happens in the worker processes, the data set needs to
    be passed back to the main process via callbacks, where data is updated.
    Processes do not share RAM!

    Public functions of this class used for steps and callbacks must return a
    function handle, which is stored in the Task instance. We use private
    (starting with '_') functions with the same name to for the actual
    implementations of the steps and callbacks. Note that steps are executed in
    the worker context, while callbacks are executed in the main process.

    See GitLab MR !4 or issues #24,#25 for more information.
    """

    def __init__(self, model_opts: dict) -> None:
        """Initialization routine for all models. Step and Callback function
        lists should point to public functions of this class."""
        self.model_opts: dict = model_opts
        self.steps: list[Callable] = [self.load_ddp, self.construct_ham, self.solve_ham, self.post_solve]
        self.callbacks: list[Callable] = [self.resume_ddp, self.update_ham, self.update_ddp, self.update_ddp]
        self.step_policy: list[str] = ['process', 'process', 'process', 'process']
        self.step_prio: list[int] = [3, 2, 1, 0]
        self.threads: list[int] = [1] * len(self.steps)
        self.gpu_workers : list[int] = [0] * len(self.steps)
        if 'solver' in model_opts:
            solver = model_opts['solver']
            for i, step in enumerate(self.steps):
                # this is better than self.steps.index(), in case we'd ever have no or multiple solve steps.
                if step == self.solve_ham:
                    self.threads[i] = solver.num_threads
                    self.step_policy[i] = solver.worker_type
                    self.solve_step_idx = i
                    if isinstance(solver, CupyShiftInvEigshSolver) or isinstance(solver, JaxEighSolver) or isinstance(solver, SLEPCGPUSolver):
                        self.gpu_workers[i] = 1
                if solver.num_processes == 1:
                    # Don't use parallel worker pools, if only one worker is
                    # used. This enables parallel construction of a single
                    # Hamiltonian, which starts its own process pool.
                    self.step_policy[i] = None
        self.resume_path, self.resume_step = resume_from()
        self.group_steps: bool = get_config_bool('tasks_grouped')

        # Do not run post_solve step if no observables need to be calculated
        if "obs" in model_opts or "dimful_obs" in model_opts:
            self.run_post_solve = True
        else:
            self.run_post_solve = False

    def step_grouping_wrapper(self, ddp: DiagDataPoint) -> Callable:
        """Public task generator.
                Packs function handle to model code implementation (private function with same name)
                and DiagDataPoint (parameters) into a object that can be pickled."""
        return TaskWrapper(self._step_grouping_wrapper, ddp).run

    def _step_grouping_wrapper(self, ddp: DiagDataPoint) -> Any:
        """This is just a wrapper method to call all single steps from one single
        Task and worker. It can be useful to get around issues with pickling of
        very large eivecs in some cases."""
        step = ddp.current_step - 1  # This needs to be reset to correct value after the enqueue_task() method.
        # Do all steps with callback except last pair (self.steps[step](ddp)())
        # will call the .run function of each step's TaskWrapper locally and
        # directly forward the result into the callback function.
        # The advantage here is, that the single TaskWrappers don't need to be
        # pickled, but only the TaskWrapper object of this grouping wrapper
        # method (which in most cases does not include (possibly very large)
        # eigenvectors).
        while step < len(self.steps) - 1:
            self.callbacks[step](ddp)(self.steps[step](ddp)())
            step += 1
            ddp.current_step = step  # this works, as the ddp is still in the same context as steps and callbacks
        # Do last step (callback must be handled externally in main thread):
        ddp.current_step += 1
        return self.steps[step](ddp)()

    def enqueue_task(self, ddp: DiagDataPoint, queue: TaskManager, progress_monitor: Progress | None = None) -> None:
        """Generate and enqueue a task for the next step of a given study point.
        It is only necessary to call this once per DiagDataPoint, as the task
        callback is automatically extended to enqueue the next step."""
        step = ddp.current_step if ddp.current_step is not None else 0
        if step >= len(self.steps):
            if progress_monitor is not None:
                progress_monitor.show(progress_monitor.jobs_done + 1)
            return
            # raise ValueError('No more steps left for %s' % study_point)
        if self.resume_path is None and self.steps[step] == self.load_ddp:
            step += 1  # skip load_ddp step silently if option is not requested
        ddp.current_step = step + 1
        f_step = self.steps[step]
        name = '%s: %s' % (f_step.__name__, job_monitor_k_b(ddp.k, ddp.paramval))

        def callback(*args, **kwds):
            """Use predefined callbacks, afterwards enqueue next Task"""
            cb_func = self.callbacks[step if not self.group_steps else -1]
            if cb_func is not None:
                cb_func(ddp)(*args, **kwds)
            # Skip post_solve step if not requested
            if not self.run_post_solve:
                if self.steps[step+1] == self.post_solve:
                    if progress_monitor is not None:
                        progress_monitor.show(progress_monitor.jobs_done + 1)
                    return
            return self.enqueue_task(ddp, queue, progress_monitor)

        # Create the task. It is put into queue automatically
        if self.group_steps:
            Task(queue, name='Grouped tasks: %s' % job_monitor_k_b(ddp.k, ddp.paramval),
                 worker_func=self.step_grouping_wrapper(ddp), callback_func=callback,
                 worker_type=self.step_policy[self.solve_step_idx],
                 n_threads=self.threads[self.solve_step_idx],
                 gpu_workers=self.gpu_workers[self.solve_step_idx],
                 priority=(0, 0 if ddp.grid_index is None else ddp.grid_index))
        else:
            Task(queue, name=name, worker_func=f_step(ddp), callback_func=callback,
                 worker_type=self.step_policy[step], n_threads=self.threads[step],
                 gpu_workers=self.gpu_workers[step],
                 priority=(self.step_prio[step], 0 if ddp.grid_index is None else ddp.grid_index))
        return

    def load_ddp(self, ddp: DiagDataPoint) -> Callable:
        """Public task generator.
        Packs function handle to model code implementation (private function
        with same name) and DiagDataPoint (parameters) into a object that can be
        pickled."""
        return TaskWrapper(self._load_ddp, ddp).run

    def _load_ddp(self, ddp: DiagDataPoint) -> Callable | None:
        """Actual (private) step implementation for loading temporary saved
        DiagDataPoint data from file. Since ddp.current_step is also read from
        file, steps that have already passed for this DiagDataPoint instance
        will be skipped automatically. Runs on worker process/thread."""
        if self.resume_path is not None:
            return resume.load_ddp_tempfile(ddp, self.resume_path, verbose=sysargv.verbose)
        else:
            return None

    def resume_ddp(self, ddp: DiagDataPoint) -> Callable:
        """Public function handle generator for a callback function.
        Callback after loading a DiagDataPoint instance"""
        def resume_callback(loaded_ddp):
            """Actual (private) callback implementation for updating a
            DiagDataPoint instance, if it has been loaded from file
            successfully. Runs in main thread."""
            if loaded_ddp is not None:
                if self.resume_step is not None:
                    # Manually overwrite step to resume from, if requested.
                    loaded_ddp.current_step = self.resume_step
                # Do the normal update procedure
                self.update_ddp(ddp)(loaded_ddp)

        return resume_callback

    def _construct_ham(self, ddp: DiagDataPoint, **ignored_opts) -> Callable:
        """Actual implementation (private) of Hamiltonian construction.
        To be overwritten by child classes."""
        raise NotImplementedError("Class %s has no 'construct_ham' step." % self.__class__.__name__)

    def construct_ham(self, ddp: DiagDataPoint) -> Callable:
        """Public task generator. Step: Hamiltonian construction."""
        return TaskWrapper(self._construct_ham, ddp, **self.model_opts, **ddp.opts).run

    def update_ham(self, ddp: DiagDataPoint) -> Callable:
        """Public function handle generator for a callback function.
        Callback after construction of Hamiltonian."""
        def update(ham):
            """Actual implementation. Updates the DiagDataPoint instance in the main thread."""
            ddp.ham = ham

        return update

    def _solve_ham(self, ddp: DiagDataPoint, solver: DiagSolver | None = None, **ignored_opts) -> DiagDataPoint:
        """Actual implementation (private) of diagonalization of Hamiltonian.
        Default minimal solve implementation. Solver argument is taken from
        modelopts dict."""
        eival, eivec = solver.solve(ddp.ham)
        return DiagDataPoint(ddp.k, eival, eivec, paramval=ddp.paramval)

    def solve_ham(self, ddp: DiagDataPoint) -> Callable:
        """Public task generator. Step: Diagonalization of Hamiltonian."""
        return TaskWrapper(self._solve_ham, ddp, **self.model_opts, **ddp.opts).run

    def _post_solve(self, ddp: DiagDataPoint, **ignored_opts) -> Callable:
        """Actual implementation (internal) of DiagDataPoint post-solve processing.
        To be overwritten by child classes."""
        raise NotImplementedError("Class %s has no 'post_solve' step." % self.__class__.__name__)

    def post_solve(self, ddp: DiagDataPoint) -> Callable:
        """Public task generator. Step: DiagDataPoint post-solve processing."""
        return TaskWrapper(self._post_solve, ddp, **self.model_opts, **ddp.opts).run

    def update_ddp(self, ddp: DiagDataPoint) -> Callable:
        """Public function handle generator for a callback function.
        Callback after changes in the DiagDataPoint instance."""
        def update(new_ddp):
            """Actual implementation. Updates the DiagDataPoint instance in the main thread."""
            ddp.update(new_ddp)
            if self.model_opts.get('tempout', False):
                resume.save_ddp_tempfile(ddp, verbose=sysargv.verbose)

        return update


class ModelLL(ModelBase):
    """Model for LL dispersion mode - Symbolic Hamiltonian version."""

    def _construct_ham(
            self,
            ddp: DiagDataPoint,
            ll_max: int = None,
            h_sym: SymbolicHamiltonian | None = None,
            params: PhysParams | None = None,
            pot: np.ndarray | None = None,
            ll_mode: str = 'full',
            split: float = 0.0,
            lattice_reg: bool = False,
            ignorestrain: bool = False,
            axial: bool = True,
            solver: DiagSolver | None = None,
            h_sym_opts: dict | None = None,
            **ignored_opts) -> tuple[MatrixT | list[MatrixT], MatrixT, SymbolicHamiltonian]:
        """Actual (private) step implementation for Hamiltonian construction.
        Runs on worker process/thread.

        Arguments:
        ddp            DiagDataPoint instance with specific parameters.
        ll_max         Integer. Maximum LL index.
        h_sym          SymbolicHamiltonian instance. The Hamiltonian.
        params         PhysParams instance.
        pot            Array. Potential V(z) in meV as function of position.
        ll_mode        LL calculation mode: legacy, sym or full

        Following arguments apply only to ll_mode 'legacy':
        split          Float. Amount of degeneracy lifting at zero magnetic
                       field.
        lattice_reg    True or False. Whether to apply lattice regularization
                       (x -> sin x). Default set to False to match symbolic
                       modes.
        ignorestrain   True or False. If True, do not include strain terms in
	                   the Hamiltonian.
        axial          True or False. If True, apply axial approximation. If
	                   False, include non-axial terms in the Hamiltonian.
        solver         DiagSolver instance. Set hllsplit magnitude from solver
                       precision.
        h_sym_opts     Modelopts dict for per-DDP construction of symbolic
                       Hamiltonian. Only required in ll_mode 'full' or 'sym' if
                       no constant h_sym can be given.

        Returns:
        Tuple with either
        - 'full' mode: Hamiltonian and zero-field split Hamiltonian
        - 'sym' or 'legacy' mode: List of single LL Hamiltonians and None (no
          splitting correction!)
        The last element of the tuple is the symbolic Hamiltonian for this DDP,
        if a new construction was necessary.
        """
        magn = ddp.paramval
        if ll_mode in ['full', 'sym'] and h_sym is None:
            # Calculate a symbolic Hamiltonian, if required, but not given. May
            # be the case if variable in-plane magnetic fields are present and
            # no single symbolic Hamiltonian can be defined.
            h_sym = SymbolicHamiltonian(
                hm.hz_sparse_split, (params,), h_sym_opts, hmagn = False, b0 = magn)
            h_sym_return = h_sym
        else:
            h_sym_return = None
        if ll_mode == 'full':
            ham = hm.hz_sparse_ll_full(h_sym, ll_max, magn, params.norbitals)
            hllsplit = None
            if abs(magn) < 1e-6:
                ll_split = 1e-8 if solver is None or solver.dtype == np.complex128 else 1e-3
                hllsplit = ll_split * hm.hsplit_ll_full(ll_max, nz=params.nz, norb=params.norbitals)
                ham += hllsplit
            if pot is not None:
                hpot = hm.hz_sparse_pot_ll_full(params, ll_max, pot, norb=params.norbitals)
                ham += hpot
            return ham, hllsplit, h_sym_return
        elif ll_mode == 'sym':
            magnz = magn.z() if isinstance(magn, Vector) else magn
            delta_n_vec = delta_n_ll(params.norbitals, magnz)
            ham_list = []
            for n in range(-2, ll_max + 1):
                ham = h_sym.ll_evaluate(n, magn, delta_n_vec)
                if pot is not None:
                    nbands = np.count_nonzero(delta_n_vec + n >= 0)
                    hpot = hm.hz_sparse_pot(params, pot, norb=nbands)
                    ham += hpot
                ham_list.append(ham)
            return ham_list, None, h_sym_return
        elif ll_mode == 'legacy':
            ham_list = []
            for n in range(-2, ll_max + 1):
                ham = hm.hz_sparse_ll(
                    magn, n, params, lattice_reg = lattice_reg,
                    split = split if magn == 0 else 0, ignorestrain = ignorestrain,
                    axial = axial)
                if pot is not None:
                    if params.norbitals == 8:
                        nbands = 1 if n == -2 else 4 if n == -1 else 7 if n == 0 else 8
                    else:
                        nbands = 1 if n == -2 else 3 if n == -1 else 5 if n == 0 else 6
                    hpot = hm.hz_sparse_pot(params, pot, norb = nbands)
                    ham += hpot
                ham_list.append(ham)
            return ham_list, None, h_sym_return

    def update_ham(self, ddp: DiagDataPoint) -> Callable:
        """Public function handle generator for a callback function.
        Callback after construction of Hamiltonian.
        Overwritten parent function, as this model has an additional Hamiltonian part."""
        def update(ham):
            """Actual implementation. Updates the DiagDataPoint instance in the main thread."""
            ddp.ham, ddp.hllsplit, ddp.h_sym = ham

        return update

    def _solve_ham(
            self,
            ddp: DiagDataPoint,
            solver: DiagSolver | None = None,
            params: PhysParams | None = None,
            ll_mode: str = 'full',
            ll_max: int| None = None,
            **ignored_opts) -> DiagDataPoint:
        """Actual (private) step implementation for Hamiltonian diagonalization.
        Runs on worker process/thread.

        Arguments:
        ddp            DiagDataPoint instance with specific parameters.
        solver		   DiagSolver instance.
        params         PhysParams instance.
        ll_mode        LL calculation mode.
        ll_max         Maximum LL in ll_mode 'full'

        Returns:
        A DiagDataPoint instance (not connected to a DiagData instance).
        """
        magn = ddp.paramval
        magnz = magn.z() if isinstance(magn, Vector) else magn
        llindex = None
        if ll_mode == 'full':
            eival, eivec = solver.solve(ddp.ham)
            # Correct for degeneracy lifting
            if abs(magn) < 1e-6:
                print("Degeneracy between Landau levels was lifted at B = %s" % magn)
                delta_eival = np.real(
                    np.array([np.vdot(eivec[:, j], ddp.hllsplit.dot(eivec[:, j])) for j in range(0, len(eival))]))
                eival -= delta_eival
            ddp.hllsplit = None  # delete split hamiltonian
            eivec = scaleup_full_eivec(eivec, params, len(eival), ll_max, magnz).T
        elif ll_mode in ['sym', 'legacy']:
            eival = []
            eivec = []
            ll_n = []
            for n, ham in enumerate(ddp.ham):  # n-2 is LL index
                eival1, eivec1 = solver.solve(ham)
                eival.extend(eival1)
                eivec.extend(scaleup_eivec(eivec1, params, len(eival1), n-2, magnz))
                ll_n.extend(np.full(len(eival1), n-2))
            eival = np.array(eival)
            eivec = np.array(eivec)
            llindex = np.array(ll_n)
        else:
            eival, eivec = None, None
        # ddp.ham will be cleared in new DiagDataPoint instance
        # ddp.current_step and ddp.grid_index will be kept via ddp.update(), if they are None.
        ddp = DiagDataPoint(0.0, eival, eivec, paramval=magn)
        ddp.llindex = llindex
        return ddp

    def _post_solve(
            self,
            ddp: DiagDataPoint | None = None,
            ll_max: int | None = None,
            h_sym: SymbolicHamiltonian | None = None,
            params: PhysParams | None  = None,
            obs: list[str] | None = None,
            obs_prop=None,
            return_eivec: bool = False,
            overlap_eivec: dict | None = None,
            berry: bool = False,
            transitions: bool = False,
            transitions_range: tuple[float, float] | None = None,
            wflocations=None,
            ll_mode: str = 'full',
            **ignored_opts) -> DiagDataPoint:  # TODO: Missing type annotations
        """Actual (private) step implementation for DiagDataPoint post solve processing.
        Runs on worker process/thread.

        Arguments:
        ddp            DiagDataPoint instance with specific parameters.
        params         PhysParams instance.
        obs            List of strings or None. Observable ids of the
                       observables that will be calculated. If None or empty
                       list, do not do anything.
        obs_prop       ObservableList instance containing all observable
                       properties.
        return_eivec   True, False or None. If True, keep eigenvector data in
                       the return value (DiagDataPoint instance). If False,
                       discard them. If None, discard them only if observables
                       have been calculated.
        overlap_eivec  A dict, whose keys are the band labels (characters) and
                       values are the eigenvectors for which overlaps can be
                       calculated with the eigenvectors of this Hamiltonian.
        berry          2-tuple, True or False. If a 2-tuple of integers,
                       calculate Berry curvature for bands with indices in this
                       range. If True, calculate Berry curvature for all states.
                       If False, do not calculate Berry curvature.
        transitions    True or False, or float. If True or a float, calculate
                       optical transitions, where a float indicates the minimum
                       transition amplitude, below which the transitions are
                       discarded. If False, do not calculate transitions.
        transitions_range  2-tuple or None. If set, calculate optical
                           transitions only for states in that energy range. If
                           None, do not restrict to an energy range.
        wflocations    List, array, or VectorGrid instance. Contains the
                       magnetic field values where wave functions should be
                       saved (plot and table). None if no wave functions should
                       be saved.
        ll_mode        LL calculation mode.

        Returns:
        A DiagDataPoint instance (not connected to a DiagData instance).
        """
        magn = ddp.paramval
        magnz = magn.z() if isinstance(magn, Vector) else magn

        ddp.calculate_observables(
            params, obs, obs_prop=obs_prop, overlap_eivec=overlap_eivec,
            magn=magn, ll_full=(ll_mode == 'full')
        )

        if ll_mode in ['full', 'sym'] and h_sym is None:
            h_sym = ddp.h_sym

        if ll_mode != 'legacy':
            if berry:
                which = berry if isinstance(berry, tuple) else None
                if magn == 0.0:
                    ddp.set_observable_value('chern', None, 0.0)
                    ddp.set_observable_value('chernsim', None, 0.0)
                else:
                    func_handle = chernnumber_ll_full if ll_mode == 'full' else chernnumber_ll
                    bc_val, bc_ei, bc_ll = func_handle(ddp, magn, h_sym, ll_max, which=which, norb=params.norbitals)
                    ddp.set_observable_value('chern', bc_ei, np.asarray(bc_val))
                    ddp.set_observable_value('chernsim', None, 1.0)

            if transitions:
                ampmin = transitions if isinstance(transitions, (float, np.floating)) else None
                func_handle = get_transitions_full if ll_mode == 'full' else get_transitions
                td = func_handle(ddp, magn, h_sym, which=transitions_range, ampmin=ampmin, norb=params.norbitals, nll=ll_max + 3)
                td.sort(in_place=True, llsort=(ll_mode != 'full'))
                ddp.transitions = td

        if isinstance(wflocations, (list, np.ndarray, VectorGrid)):
            wfmagn = magn if isinstance(magn, Vector) else Vector(magn, astype='z')
            if locations_index(wflocations, wfmagn, vec_numeric = magnz) is not None:
                return_eivec = True

        if ddp.h_sym is not None:
            # Clean-up temporary symbolic hamiltonian per DDP, as it is not required any more.
            del ddp.h_sym
        save_ddp = get_config('diag_save_binary_ddp')
        if save_ddp in ['numpy', 'npz']:
            npz_filename = "ddp_%s_%s.npz" % (ddp.file_id(), ddp.hash_id())
            ddp.to_binary_file(npz_filename)
        elif save_ddp in ['hdf5', 'h5']:
            h5_filename = "ddps.h5"
            ddp.to_binary_file(h5_filename)
        if not return_eivec:
            ddp.delete_eivec()
        return ddp

class ModelMomentum1D(ModelBase):
    """Model for k 1D (dispersion) mode."""

    def _construct_ham(
            self,
            ddp: DiagDataPoint,
            params: PhysParams | None = None,
            periodicy: bool = False,
            lattice_reg: bool = False,
            split: float = 0.0,
            splittype: str = 'auto',
            ignorestrain: bool = False,
            gauge_zero: float = 0.0,
            solver: DiagSolver | None = None,
            axial: bool = True,
            pot: np.ndarray | None = None,
            bia: bool = False,
            ignore_magnxy: bool = False,
            **ignored_opts) -> MatrixT:
        """Actual (private) step implementation for Hamiltonian construction.
        Runs on worker process/thread.

        Arguments:
        ddp            DiagDataPoint instance with specific parameters.
        params         PhysParams instance.energy
        periodicy      True or False. Whether the geometry in the transversal
                       (y) direction is periodic/cylindrical (True) or finite
                       (False).
        lattice_reg    True or False. Whether to apply lattice regularization
                       (x -> sin x).
        split          Float. Amount of degeneracy lifting.
        splittype      String. Type of degeneracy lifting.
        ignorestrain   True or False. If True, do not include strain terms in
                       the Hamiltonian.
        gauge_zero     Float. Shifts the gauge field by this amount. See
                       hamiltonian/full.py.
        solver		   DiagSolver instance
        axial          True or False. If True, apply axial approximation. If
                       False, include non-axial terms in the Hamiltonian.
        pot            Array, 1-, 2- or 3-dimensional. Potential V(z) (1-dim) or
                       V(z, y) (2-dim) or V_o(z, y) (3-dim) in meV as function
                       of position and optionally orbital (3-dim only).
        bia            True or False. If True, include BIA terms in the
                       Hamiltonian.
        ignore_magnxy  True or False. If True, neglect the in-plane components
                       of the orbital part of the magnetic field. Only for
                       legacy reasons, e.g., comparing with results that were
                       calculated when these terms were not yet implemented.

        Returns:
        A sparse matrix.
        """
        b = ddp.paramval
        kx, ky = ddp.k.xy()
        if abs(ky) > 1e-6:
            sys.stderr.write("ERROR (ModelMomentum1D._construct_ham): y component of the momentum must be zero\n")
        kterms = hm.h_kterms(params, axial=axial) if params.lattice_transformed_by_matrix() else None
        if b == 0.0:
            ham = hm.hzy_sparse(
                kx, 0.0, params, periodicy=periodicy, solver=solver,
                lattice_reg=lattice_reg, ignorestrain=ignorestrain, axial=axial,
                bia=bia, kterms=kterms)
        else:
            ham = hm.hzy_sparse_magn(
                kx, b, params, periodicy=periodicy, solver=solver,
                lattice_reg=lattice_reg, ignorestrain=ignorestrain, axial=axial,
                bia=bia, kterms=kterms, gauge_zero=gauge_zero,
                ignore_magnxy=ignore_magnxy)
        if split != 0.0:
            hamsplit = split * hm.hsplit_full(params, splittype, k=[kx], bia=bia, lattice_reg=lattice_reg)
            ham += hamsplit

        if pot is not None:
            ham += hm.h_pot_1d(pot, params)
        return ham

    def _post_solve(
            self,
            ddp: DiagDataPoint,
            params: PhysParams | None = None,
            obs: list[str] | None = None,
            obs_prop=None,
            overlap_eivec: dict | None = None,
            ignorestrain: bool = False,
            axial: bool = True,
            split: float = 0.0,
            lattice_reg: bool = False,
            berry: bool = False,
            currents: bool = False,
            return_eivec: bool | None = None,
            wflocations=None,
            **ignored_opts) -> DiagDataPoint:  # TODO: Missing type annotations
        """Actual (private) step implementation for DiagDataPoint post solve processing.
        Runs on worker process/thread.

        Arguments:
        ddp            DiagDataPoint instance with specific parameters.
        params         PhysParams instance.
        obs            List of strings or None. Observable ids of the
                       observables that will be calculated. If None or empty
                       list, do not do anything.
        obs_prop       ObservableList instance containing all observable
                       properties.
        overlap_eivec  A dict, whose keys are the band labels (characters) and
                       values are the eigenvectors for which overlaps can be
                       calculated with the eigenvectors of this Hamiltonian.
        ignorestrain   True or False. If True, do not include strain terms in
                       the Hamiltonian.
        axial          True or False. If True, apply axial approximation. If
                       False, include non-axial terms in the Hamiltonian.
        split          Float. Amount of degeneracy lifting.
        lattice_reg    True or False. Whether to apply lattice regularization
                       (x -> sin x).
        berry          2-tuple, True or False. If a 2-tuple of integers,
                       calculate Berry curvature for bands with indices in this
                       range. If True, calculate Berry curvature for all states.
                       If False, do not calculate Berry curvature.
        currents       String, True or False. If True, calculate the expectation
                       values of dH/dk_x and dH/dk_y. If a string, that
                       corresponds to a valid observable O, also calculate those
                       of the symmetrized products {dH/dk_i, O}/2.
        return_eivec   True, False or None. If True, keep eigenvector data in
                       the return value (DiagDataPoint instance). If False,
                       discard them. If None, discard them only if observables
                       have been calculated.
        wflocations    List, array, or VectorGrid instance. Contains the
                       magnetic field values where wave functions should be
                       saved (plot and table). None if no wave functions should
                       be saved.

        Returns:
        A DiagDataPoint instance.
        """
        ddp.calculate_observables(
            params, obs, obs_prop = obs_prop, overlap_eivec = overlap_eivec,
            magn = ddp.paramval)

        if berry:
            berry_dk = get_config_num('berry_dk', minval=0)
            if berry_dk == 0:
                sys.stderr.write(
                    "ERROR (diagonalization.hz): Berry curvature momentum step must be a positive number.\n")
                raise ValueError
            which = berry if isinstance(berry, tuple) else None
            bc_val, bc_ei, _ = berrycurv_k(
                ddp, hm.hz_sparse_split, params, dk=berry_dk, which=which,
                lattice_reg=lattice_reg, split=split, ignorestrain=ignorestrain,
                axial=axial)
            ddp.set_observable_value('berry', bc_ei, np.asarray(bc_val))
            ibc_val = ddp.get_observable('berry') * ddp.get_observable('isopz')
            ddp.set_observable_value('berryiso', np.arange(0, ddp.neig), ibc_val)

        if currents:
            # We need a symbolic version of the Hamiltonian H in order to
            # calculate dH/dk_i. Due to the different argument pattern of
            # self._construct_ham() compared to SymbolicHamiltonian.__init__(),
            # we define a wrapper function.
            def ham_wrapper(k, b0, *args, **kwds):
                empty_ddp = DiagDataPoint(Vector(*k), paramval=b0)
                return self._construct_ham(*args, ddp=empty_ddp, **kwds)
            hsym = SymbolicHamiltonian(ham_wrapper, args=(), kwds=self.model_opts)
            set_disp_derivatives_dhdk(hsym, ddp, dhdk=True, v=True)

            if isinstance(currents, str):
                currents_obsid = currents
                set_disp_derivatives_dhdk_obs(hsym, ddp, currents_obsid, params, dhdk=True, v=True)

        # Wave functions
        if isinstance(wflocations, (list, np.ndarray, VectorGrid)):
            if locations_index(wflocations, ddp.k) is not None:
                return_eivec = True

        save_ddp = get_config('diag_save_binary_ddp')
        if save_ddp in ['numpy', 'npz']:
            npz_filename = "ddp_%s_%s.npz" % (ddp.file_id(), ddp.hash_id())
            ddp.to_binary_file(npz_filename)
        elif save_ddp in ['hdf5', 'h5']:
            h5_filename = "ddps.h5"
            ddp.to_binary_file(h5_filename)
        if get_config_bool('diag_save_binary_ddp_delete_eivec') and ddp.binary_file is not None:
            return_eivec = False
        if return_eivec is None:
            return_eivec = (obs is None or obs == [])
        if not return_eivec:
            ddp.delete_eivec()
        return ddp

class ModelMomentum2D(ModelMomentum1D):
    """Model for k 2D (dispersion) mode.
    Only differs from 1D model in Hamiltonian construction step.
    """

    def _construct_ham(
            self,
            ddp: DiagDataPoint,
            params: PhysParams | None = None,
            lattice_reg: bool = False,
            split: float = 0.0,
            splittype: str = 'auto',
            ignorestrain: bool = False,
            solver: DiagSolver | None = None,
            axial: bool = True,
            pot: np.ndarray | None = None,
            bia: bool = False,
            ignore_magnxy: bool = False,
            **ignored_opts) -> MatrixT:
        """Actual (private) step implementation for Hamiltonian construction.
        Runs on worker process/thread.

        Arguments:
        ddp            DiagDataPoint instance with specific parameters.
        params         PhysParams instance.energy
        lattice_reg    True or False. Whether to apply lattice regularization
                       (x -> sin x).
        split          Float. Amount of degeneracy lifting.
        splittype      String. Type of degeneracy lifting.
        ignorestrain   True or False. If True, do not include strain terms in the
                       Hamiltonian.
        solver		   DiagSolver instance
        axial          True or False. If True, apply axial approximation. If False,
                       include non-axial terms in the Hamiltonian.
        pot            Array, 1- or 2-dimensional. Potential V(z) (1-dim) or V(z, y)
                       (2-dim) in meV as function of position.
        bia            True or False. If True, include BIA terms in the Hamiltonian.
        ignore_magnxy  True or False. If True, neglect the in-plane components of
        		       the orbital part of the magnetic field. Only for legacy
        		       reasons, e.g., comparing with results that were calculated
        		       when these terms were not yet implemented.

        Returns:
        A sparse matrix.
        """
        b = ddp.paramval
        kx, ky = ddp.k.xy()
        kterms = hm.h_kterms(params, axial=axial) if params.lattice_transformed_by_matrix() else None
        ham = hm.hz_sparse(
            [kx, ky], b, params, solver=solver, lattice_reg=lattice_reg,
            ignorestrain=ignorestrain, axial=axial, bia=bia, kterms=kterms,
            ignore_magnxy=ignore_magnxy)
        if split != 0.0:
            hamsplit = split * hm.hsplit_full(params, splittype, k=[kx, ky], bia=bia, lattice_reg=lattice_reg)
            ham += hamsplit
        if pot is not None:
            hpot = hm.hz_sparse_pot(params, pot)
            ham += hpot
        return ham
