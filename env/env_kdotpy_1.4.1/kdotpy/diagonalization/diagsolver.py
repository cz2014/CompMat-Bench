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

from sys import stderr
from os import environ
from time import sleep, perf_counter as rtime  # high precision timing
from typing import Any

from ..config import get_config, get_config_int, get_config_num_auto
from ..types import MatrixT, SolveResultT, DiagSolver
from .. import cmdargs
from .. import intervaltools

from .diagdata import DiagDataPoint, NoOverlapError, SubsetError

# Previously, we set the thread number for scipy / umfpack / pyMKL before
# importing. (For Intel MKL / FEAST it's ok to reset this later.) We changed to
# using the threadpoolctl package in order to be able to set the number of
# threads also if one uses 'import kdotpy; kdotpy.run(*args)'.
# The other environment variables below need not be changed dynamically.

environ['CUPY_CUDA_PER_THREAD_DEFAULT_STREAM'] = '1'  # Use a separate CUDA stream for each CPU thread (NOT process!)
environ['KMP_WARNINGS'] = 'False'  # suppress deprecation warnings from MKL via pyMKL

environ["JAX_ENABLE_X64"] = "True"  # If JAX is used, make sure it runs with double precision

# Numpy/Scipy brings the default math, sparse matrix tools and solvers
# UMFPACK is loaded automatically for LU solve, if installed.
import numpy as np
from scipy.linalg import eigh
from scipy.sparse import eye, issparse, csr_matrix
from scipy.sparse.linalg import eigsh, factorized, splu, LinearOperator

# Optional packages for additional solver configuration. Only loaded if
# installed in environment. Since they are optional, any import errors are
# suppressed.

# Interface to slepc
try:
    import petsc4py
    # add "-log_view", "-info" for more output
    petsc4py.init(["-use_gpu_aware_mpi", "0"])
    from petsc4py import PETSc
    from slepc4py import SLEPc
    if(PETSc.ScalarType == np.complex128):
        HAS_SLEPC = True
    else:
        HAS_SLEPC = False
except Exception:
    HAS_SLEPC = False

# CUPY package for Nvidia CUDA support.
# Best performance with large Hamiltonians and GPUs featuring TensorCores.
try:
    import cupy
    from cupyx.scipy.sparse import csr_matrix as cupy_csr_matrix
    from cupyx.scipy.sparse.linalg import eigsh as cupy_eigsh, factorized as cupy_factorized
    from cupyx.scipy.sparse.linalg import LinearOperator as CupyLinOp
    HAS_CUPY = True
except:
    HAS_CUPY = False

try:
    import jax.numpy as jnp
    import jax
    HAS_JAX = True
except Exception:
    HAS_JAX = False

# Interface to Intel MKL PARDISO solver (for LU decomposition - shift-invert solve)
try:
    # Unfortunately, pyMKL is no longer maintained (as of v0.0.3)
    # See PR #16 for added single precision and new Intel OneAPI support.
    # Installation of an suitably updated version is possible via:
    # 'pip install git+git://github.com/fjbay/pyMKL@patch-1#egg=pyMKL'
    # Alternative: pypardiso, but has more overhead per call and does not support complex matrices.
    # When running on AMD CPUs, pay special attention to the MKL version
    # you are using (see e.g. wiki tutorials/eigensolver optimization).
    import pyMKL
    HAS_PARDISO = True
except:
    HAS_PARDISO = False

# Interface to FEAST solver libraries (either custom compilation of v4 or v2 via
# Intel MKL). This algorithm does not require a shift-invert strategy, but finds
# all eigenstates in a given energy interval, making it more efficient in
# theory. Practically, with typical eigenvalue problems for kdotpy, it offers
# worse performance due to dense eigenstate clustering at the cut-off energies.
try:
    from . import feastsolver as fs  # this prepares the solver for execution
    HAS_FEAST = True
except:
    HAS_FEAST = False

# global variables: issue warnings enabled
feast_not_found_warning = True
no_pardiso_warning = True
no_cupy_warning = True
cupy_warning_2d = True
no_jax_warning = True
no_slepc_warning = True


class FastSpLuInv(LinearOperator):
    """
    SpLuInv:
       helper class to repeatedly solve M*x = rhs
       using a sparse LU-decomposition of mat M
    """

    def __init__(self, mat: MatrixT, solver: str = 'umfpack') -> None:
        t0 = rtime()
        super().__init__(mat.dtype, mat.shape)
        if solver == 'umfpack':
            self.M_lu_solve = factorized(mat)  # Falls back to SuperLU silently if Scikit.umfpack is not available
        elif solver == 'superlu':
            # Chooses SuperLU directly. Equals default eigs(h) behaviour,
            # but gives enables us to measure timing statistics.
            self.M_lu_solve = splu(mat).solve
        elif solver == 'pardiso':
            pardiso_invert = pyMKL.pardisoSolver(mat, mtype=-4)
            pardiso_invert.factor()
            self.M_lu_solve = pardiso_invert.solve
            self.solver_handle = pardiso_invert
        else:
            raise ValueError('Solver %s not supported' % solver)
        self.solver: str = solver
        self.isreal: bool = not np.issubdtype(self.dtype, np.complexfloating)
        self.n_mult: int = 0
        self.pref_time: float = rtime()
        self.total_t_arpack: int = 0
        self.total_t_lu_solve: int = 0
        self.verbose: bool = cmdargs.sysargv.verbose
        if self.verbose:
            stderr.write("Initialized and factorized %s LU solver in %.3g s.\n" % (solver, rtime()-t0))

    def __del__(self) -> None:
        try:
            if self.solver == 'pardiso':
                self.solver_handle.clear()  # Clear memory of pardiso instance
            if self.verbose:
                stderr.write('Total ARPACK %.3g s (%.3g %%); Total LU-Inv solve %.3g s. Used %d MatVec operations.\n'
                             % (self.total_t_arpack, 100 * self.total_t_arpack / (self.total_t_lu_solve + self.total_t_arpack),
                                self.total_t_lu_solve, self.n_mult))
        except AttributeError:
            # In some rare cases (e.g. memory allocation errors), this instance
            # will not have all (valid) attributes. This would throw additional
            # distracting secondary error messages, therefore, we suppress those
            # errors completely. The verbose output might also be suppressed, but
            # would not be of much use in this case anyway.
            pass

    def _matvec(self, rhs: MatrixT) -> MatrixT:
        """Custom definition of matrix-vector product M^-1 * rhs, where
        the matrix M^-1 would be the virtual inverse of the original M (not actually calculated)
        and the Vector rhs (usually an eigenvector candidate).
        """
        rhs = np.asarray(rhs)
        t0 = rtime()

        # careful here: splu.solve will throw away imaginary
        # part of rhs if M is real (this part is taken from the original scipy definition)
        if self.solver != 'pardiso' and self.isreal and np.issubdtype(rhs.dtype, np.complexfloating):
            x = (self.M_lu_solve(np.real(rhs).astype(self.dtype)) + 1j * self.M_lu_solve(np.imag(rhs).astype(self.dtype)))
        else:
            x = self.M_lu_solve(rhs.astype(self.dtype))

        # Extremely verbose output for debugging:
        # stderr.write('MatVecMul %d done (%.3f ms). %.3f ms between calls\n'
        #              % (self.n_mult, (rtime() - t0) * 1000, (t0 - self.pref_time) * 1000))

        # Time between calls is about 99% due to ARPACK for standard hermitian eigenvalue problem:
        self.total_t_arpack += t0 - self.pref_time
        self.total_t_lu_solve += rtime() - t0
        self.n_mult += 1  # Count matvec multiplications = ARPACK.iterations
        self.pref_time = rtime()
        return x


class DiagSparseSolver:
    """Template class for sparse matrix eigensolvers."""
    def __init__(self,
            num_processes: int,
            num_threads: int,
            neig: int,
            worker_type: str | None = None,
            **ignored_opts) -> None:
        self.neig: int = neig
        self.num_processes: int = num_processes
        self.num_threads: int = num_threads
        self.lasteivec: MatrixT = None
        self.reuse_eivec: bool = False
        self.verbose: bool = cmdargs.sysargv.verbose
        self.dtype: np.dtype = np.complex128
        self.eival_accuracy: float = 1e-6
        self.worker_type: str = worker_type
        # The value handle_sigchld tells the TaskManager to redefine SIGCHLD to
        # terminate on that signal. This is needed to make multiprocessing
        # handle the case that a child process dies in a graceful way. However,
        # this can interfere with some (external) solvers, like jax. For those
        # solvers, handle_sigchld should be set to False. The value does not
        # affect behaviour on Windows.
        self.handle_sigchld: bool = True
        # Previously, we set the number of threads here. Currently, the
        # threadpoolctl package takes care of this in tasks.py and parallel.py.

    def solve(self, mat: MatrixT) -> SolveResultT:
        """Solves the configured eigenvalue problem for matrix given by 'mat'.
        Returns: (eigenvalues, eigenvectors)."""
        raise NotImplementedError("Class 'DiagSparseSolver' has no implementation for 'solve'. Use a child class instead.")


class EighSolver(DiagSparseSolver):
    def __init__(self, num_processes: int, num_threads: int, **kwds) -> None:
        super().__init__(num_processes, num_threads, 0, **kwds)

    def solve(self, mat: MatrixT) -> SolveResultT:
        dense_mat = mat.toarray() if issparse(mat) else mat
        eival, eivec = eigh(dense_mat)
        return eival, eivec


class FeastSolver(DiagSparseSolver):
    def __init__(self, num_processes: int, num_threads: int, neig: int, minval: float, maxval: float, **kwds) -> None:
        super().__init__(num_processes, num_threads, neig, **kwds)
        self.minval: float = minval
        self.maxval: float = maxval

    def solve(self, mat: MatrixT) -> SolveResultT:
        max_tries = 4
        eival, eivec = None, None
        for i in range(max_tries):
            if i > 0:
                stderr.write("Retrying: %d/%d\n" % (i + 1, max_tries))
            if not self.reuse_eivec:
                self.lasteivec = None
            eival, eivec, lasteivec, info = fs.feastsolve(mat, self.minval, self.maxval, self.neig, self.lasteivec,
                                                          verbose=self.verbose)
            if info == 0:
                # Optimization for best speed: Set the requested amount of
                # eigenvalues for following iterations (feastsolver
                # automatically uses a optimal subspace size). Only increase if
                # there are substantially more eigenvalues (as this triggers a
                # reset of the eigenvector subspace).
                if len(eival) < self.neig or len(eival) > 1.1 * self.neig:
                    self.neig = len(eival)
                    stderr.write("Info (FeastSolver): Automatically resizing subspace for optimal performance. "
                                 "New number of requested eigenstates for next use of this solver: %d.\n" % self.neig)
                self.lasteivec = lasteivec
                return eival, eivec
            if info == 1:
                stderr.write("Warning (FeastSolver): No eigenvalue found in energy range.\n")
                return None, None
            if info == 2:
                stderr.write("Warning (FeastSolver): Did not converge. This can have multiple issues. "
                             "Try adjusting the energy range or requested eigenvalues.\n")
                self.lasteivec = None  # Reset the input eivec for safety
                break
            if info == 3:
                stderr.write("Warning (FeastSolver): The amount of requested eigenvalues is too small. "
                             "Automatically doubling requested eigenvalues for following calculations.\n")
                self.neig *= 2
                stderr.write("New value: %d.\n" % self.neig)
            # All other flags are ignored or raise an error anyway.
        stderr.write("Warning (FeastSolver): Output eigenstates are unreliable. Please check.\n")
        return eival, eivec


class EigshSolver(DiagSparseSolver):
    """Implements the default scipy eigsh solver."""
    def __init__(self, num_processes: int, num_threads: int, neig: int, targetval: float, **kwds) -> None:
        super().__init__(num_processes, num_threads, neig, **kwds)
        self.targetval: float = targetval

    def solve(self, mat: MatrixT) -> SolveResultT:
        eival, eivec = eigsh(mat, self.neig, sigma=self.targetval, v0=self.lasteivec if self.reuse_eivec else None)
        if self.reuse_eivec:
            self.lasteivec = eivec[:, 0]
        return eival, eivec

def get_params_hermitian_complex(n: int, nev: int, mpd_max: int = 600) -> tuple[int, int]:
    """Heuristic finding of dimensions for Shift-Invert + Krylov-Schur.

    For more information, see the documentation of SLEPc and slepc4py:
    https://slepc.upv.es/release/documentation/manual/eps.html
    https://slepc.upv.es/release/slepc4py/reference/slepc4py.SLEPc.EPS.html

    Arguments:
    n        Integer. Matrix dimension.
    nev      Integer. Number of eigenvalues to compute.
    mpd_max  Integer. Upper limit for the maximum projected dimension

    Returns:
    ncv   Integer. Maximum dimension of the subspace to be used by the solver,
          i.e., number of column vectors.
    mpd   Integer. Maximum dimension allowed for the projected problem.
    """
    # ---- ncv ----
    if nev < 50:
        ncv = 4 * nev
    elif nev < 200:
        ncv = 3 * nev
    else:
        ncv = 2 * nev

    # mpd < mpd_max
    mpd = min(ncv, mpd_max)

    # ncv <= n or ncv <= nev + mpd
    ncv = min(ncv, n, nev + mpd)

    return ncv, mpd

class SLEPCSolver(DiagSparseSolver):
    """Implements the eigensolver from SLEPc (CPU implementation)"""
    def __init__(self, num_processes: int, num_threads: int, neig: int, targetval: float, slepc_opts: str = "", **kwds) -> None:
        super().__init__(num_processes, num_threads, neig, **kwds)
        self.targetval: float = targetval
        self.slepc_opts: str = slepc_opts

    def solve(self, mat: MatrixT) -> SolveResultT:
        n = mat.shape[0]
        sigma = self.targetval

        opts = PETSc.Options()
        # Default options
        # Mumps
        opts["st_mat_mumps_icntl_13"] = 1  # compute the inertia
        opts["st_mat_mumps_icntl_28"] = 2  # parallel ordering
        opts["st_mat_mumps_icntl_29"] = 2  # parallel analysis
        # LU Solver
        # opts["st_pc_factor_mat_ordering_type"] = "nd"  # Sets the ordering routine
        # read user input
        opts.insertString(self.slepc_opts)

        # 1) Create PETSc-Matrix
        A_csr = csr_matrix(mat)
        A = PETSc.Mat().create(comm=PETSc.COMM_SELF)
        # Set hermitian flag
        A.setOption(PETSc.Mat.Option.HERMITIAN, True)
        A.setSizes(A_csr.shape)
        # Set Matrix to be Compressed sparse row
        # Can be overriten with -mat_type
        A.setType(PETSc.Mat.Type.AIJ)
        A.setFromOptions()
        A.setPreallocationCSR((A_csr.indptr, A_csr.indices, None))
        # Set up the internal data structures for using the matrix.
        A.setUp()
        # Fill Matrix
        A.setValuesCSR(A_csr.indptr, A_csr.indices, A_csr.data)
        A.assemble()

        # 2) Setup EPS Solver
        eps = SLEPc.EPS().create()
        # Set A (kp-Hamiltonian) as the eigenvalue problem.
        eps.setOperators(A)
        # Hermitian eigenproblem
        eps.setProblemType(SLEPc.EPS.ProblemType.HEP)
        # Set the Krylov-Schur as the solver to be used.
        eps.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
        # Sought the spectrum closest to the real part of the target.
        eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_REAL)
        # Set target value (targetenergy).
        eps.setTarget(sigma)

        # Get the spectral transformation
        st = eps.getST()
        # Shift-and-invert as spectral transformation.
        st.setType(SLEPc.ST.Type.SINVERT)
        # KSP => This object manages the linear solves in PETSc.
        ksp = st.getKSP()
        # Applies ONLY the preconditioner exactly once.
        # Can also be set via -st_ksp_type
        ksp.setType(PETSc.KSP.Type.PREONLY)
        # Return the preconditioner.
        pc = ksp.getPC()
        # Set the preconditioner Type
        # See also: https://petsc.org/release/overview/linear_solve_table/
        # Can also be set via -st_pc_type
        pc.setType(PETSc.PC.Type.LU)
        # Set the solver package used to perform the factorization.
        # Can also be set via -st_pc_factor_mat_solver_type
        pc.setFactorSolverType(PETSc.Mat.SolverType.MUMPS)
        # nev: Number of eigenvectors
        # ncv: Number of column vectors to be used by the solution algorithm
        # mpd: Maximum projected dimension
        # The SLEPC docs write: "Note that ncv must be at least equal to nev,
        # although in general it is recommended (depending on the method) to
        # work with a larger subspace, for instance ncv >= 2 * nev or even more"
        # See:
        # https://slepc.upv.es/release/documentation/manual/eps.html
        # https://slepc.upv.es/release/slepc4py/reference/slepc4py.SLEPc.EPS.html
        ncv, mpd = get_params_hermitian_complex(n, self.neig)
        eps.setDimensions(nev=self.neig, ncv=ncv, mpd=mpd)

        # 3) Set Initial Eigen Space
        if self.lasteivec is not None:
            X = self.lasteivec
            petsc_vecs = [
                PETSc.Vec().createWithArray(X[:, j], comm=PETSc.COMM_SELF)
                for j in range(X.shape[1])
            ]
            eps.setInitialSpace(petsc_vecs)

        # 4) Set Options from CMD
        eps.setFromOptions()

        # 5) Solve Eigenvalues
        eps.solve()
        nconv = eps.getConverged()

        # 6) Collect Eigenvalues
        nev = min(nconv, self.neig)

        evals = np.empty(nev, dtype=np.float64)
        evecs = np.empty((n, nev), dtype=np.complex128)

        vr = A.createVecRight()
        for i in range(nev):
            lam = eps.getEigenpair(i, vr)
            evals[i] = lam
            evecs[:, i] = vr.getArray()

        vr.destroy()

        # Sort Eigenvalues
        idx = np.argsort(np.abs(evals - sigma))
        eigvals = evals[idx][:self.neig]
        eigvecs = evecs[:, idx][:, :self.neig]

        eps.destroy()
        A.destroy()

        if self.reuse_eivec:
            self.lasteivec = eigvecs
        return eigvals.astype(np.double), eigvecs.astype(np.complex128)


class EigshMultiSolver(DiagSparseSolver):
    """Implements the default scipy eigsh solver."""
    def __init__(self, num_processes: int, num_threads: int, neig: int, targetval: list[float], **kwds) -> None:
        super().__init__(num_processes, num_threads, neig, **kwds)
        self.targetval: list[float] = targetval

    def solve(self, mat: MatrixT) -> SolveResultT:
        if len(self.targetval) == 0:
            raise ValueError
        eival, eivec = eigsh(mat, self.neig, sigma=self.targetval[0])
        intervals = [intervaltools.from_eivals(eival, self.targetval[0])]
        if len(self.targetval) == 1:
            return eival, eivec
        ddp = DiagDataPoint(0, eival, eivec).sort_by_eival(inplace=True)
        for targetval in self.targetval[1:]:
            eival, eivec = eigsh(mat, self.neig, sigma=targetval)
            ddp.extend_by(0, eival, eivec).sort_by_eival(inplace=True)
            intervals.append(intervaltools.from_eivals(eival, targetval))
        intervals = intervaltools.normalize(intervals)

        if len(intervals) > 1:
            stderr.write("ERROR (EigshMultiSolver.solve): Disconnected eigenvalue ranges: " + ", ".join(["[%.3f, %.3f]" % i for i in intervals]) + ".\n")
            exit(1)  # TODO: Handle this exception more gently?
        return ddp.eival, ddp.eivec

class CustomShiftInvEigshSolver(EigshSolver):
    """Implements the default scipy eigsh solver with a configurable shift-invert factorization."""
    def __init__(self, num_processes: int, num_threads: int, neig: int, targetval: float, shift_invert_solver: str, **kwds) -> None:
        super().__init__(num_processes, num_threads, neig, targetval, **kwds)
        self.shift_invert_solver: str = shift_invert_solver

    def solve(self, mat: MatrixT) -> SolveResultT:
        mat -= self.targetval * eye(mat.shape[0])
        shift_invert_factorization = FastSpLuInv(mat, solver=self.shift_invert_solver)
        eival, eivec = eigsh(mat, self.neig, sigma=self.targetval, v0=self.lasteivec, OPinv=shift_invert_factorization)
        if self.reuse_eivec:
            self.lasteivec = eivec[:, 0]
        return eival, eivec

class SLEPCGPUSolver(CustomShiftInvEigshSolver):
    """Implements the eigensolver from SLEPc (GPU implementation)"""
    def __init__(self, diag_solver_slepc_options: str = "", *args, **kwds) -> None:
        super().__init__(*args, **kwds)
        self.diag_solver_slepc_options: str = diag_solver_slepc_options

    def solve(self, mat: MatrixT) -> SolveResultT:
        n = mat.shape[0]
        sigma = self.targetval

        # Evaluate Options String
        opts = PETSc.Options()
        opts.insertString(self.diag_solver_slepc_options)

        # 1) Create PETSc-Matrix
        A_csr = csr_matrix(mat)
        A = PETSc.Mat().create(comm=PETSc.COMM_SELF)
        # Set hermitian flag
        A.setOption(PETSc.Mat.Option.HERMITIAN, True)
        A.setSizes(A_csr.shape)
        # Set Matrix to be Compressed sparse row
        # Can be overriten with -mat_type
        A.setType(PETSc.Mat.Type.AIJCUSPARSE)
        A.setFromOptions()
        A.setPreallocationCSR((A_csr.indptr, A_csr.indices, None))
        A.setUp() # Set up the internal data structures for using the matrix.
        # Fill Matrix
        A.setValuesCSR(A_csr.indptr, A_csr.indices, A_csr.data)
        A.assemble()

        # 2) Setup EPS Solver
        eps = SLEPc.EPS().create()
        # Set A (kp-Hamiltonian) as the eigenvalue problem.
        eps.setOperators(A)
        # Hermitian eigenproblem
        eps.setProblemType(SLEPc.EPS.ProblemType.HEP)
        # Set the Krylov-Schur as the solver to be used.
        eps.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
        # Sought the spectrum closest to the real part of the target.
        eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_REAL)
        # Set target value (targetenergy).
        eps.setTarget(sigma)

        # Get the spectral transformation
        st = eps.getST()
        # Shift-and-invert as spectral transformation.
        st.setType(SLEPc.ST.Type.SINVERT)
        # KSP => This object manages the linear solves in PETSc.
        ksp = st.getKSP()
        # Generalized Minimal Residual method with restart
        # Can also be set via -st_ksp_type
        ksp.setType(PETSc.KSP.Type.GMRES)
        # Return the preconditioner.
        pc = ksp.getPC()
        # Set the preconditioner Type
        # See also: https://petsc.org/release/overview/linear_solve_table/
        # Can also be set via -st_pc_type
        pc.setType(PETSc.PC.Type.LU)
        # Set the solver package used to perform the factorization.
        # Can also be set via -st_pc_factor_mat_solver_type
        pc.setFactorSolverType(PETSc.Mat.SolverType.CUSPARSE)

        # nev: Number of eigenvectors
        # ncv: Number of column vectors to be used by the solution algorithm
        # mpd: Maximum projected dimension
        # The SLEPC docs write: "Note that ncv must be at least equal to nev,
        # although in general it is recommended (depending on the method) to
        # work with a larger subspace, for instance ncv >= 2 * nev or even more"
        # See:
        # https://slepc.upv.es/release/documentation/manual/eps.html
        # https://slepc.upv.es/release/slepc4py/reference/slepc4py.SLEPc.EPS.html
        ncv, mpd = get_params_hermitian_complex(n, self.neig, 1000)
        eps.setDimensions(nev=self.neig, ncv=ncv, mpd=mpd)

        # 3) Set Initial Eigen Space
        if self.lasteivec is not None:
            X = self.lasteivec
            petsc_vecs = [
                PETSc.Vec().createWithArray(X[:, j], comm=PETSc.COMM_SELF)
                for j in range(X.shape[1])
            ]
            eps.setInitialSpace(petsc_vecs)

        # 4) Set Options from CMD
        eps.setFromOptions()

        # 5) Solve Eigenvalues
        eps.solve()
        nconv = eps.getConverged()

        # 6) Collect Eigenvalues
        nev = min(nconv, self.neig)

        evals = np.empty(nev, dtype=np.float64)
        evecs = np.empty((n, nev), dtype=np.complex128)

        vr = A.createVecRight()
        for i in range(nev):
            lam = eps.getEigenpair(i, vr)
            evals[i] = lam
            evecs[:, i] = vr.getArray()

        vr.destroy()

        # Sort Eigenvalues
        idx = np.argsort(np.abs(evals - sigma))
        eigvals = evals[idx][:self.neig]
        eigvecs = evecs[:, idx][:, :self.neig]

        eps.destroy()
        A.destroy()

        if self.reuse_eivec:
            self.lasteivec = eigvecs
        return eigvals.astype(np.double), eigvecs.astype(np.complex128)


class CupyShiftInvEigshSolver(CustomShiftInvEigshSolver):
    """Implements the cupy version of eigsh solver running on CUDA capable GPUs
    with  a configurable shift-invert factorization."""
    def __init__(self, *args, dtype: str = 'single', **kwds) -> None:
        super().__init__(*args, **kwds)
        # Do not reuse eivecs. Tests yield no better performance,
        # just more problems (thread safety!)
        self.reuse_eivec: bool = False
        if dtype == 'single':
            self.dtype: np.dtype = np.complex64
            self.eival_accuracy: float = 1e-3
        else:
            self.dtype: np.dtype = np.complex128
        # Threshold for successful single precision gemm operation (empirical value):
        self.gemm_dim_threshold: float | None = get_config_num_auto('diag_solver_cupy_gemm_dim_thr')
        if self.gemm_dim_threshold is None:
            self.gemm_dim_threshold = 4e6

    def solve(self, mat: MatrixT) -> SolveResultT:
        """Specialized solve implementation for optimal GPU performance.
        Two algorithms are available:
        - double precision solver: Just calls the cupy eigsh method for the full problem set in double precision.
        - single precision solver: Makes best use of Nvidia GPUs' TensorCores (starting from Volta generation models).
            To prevent numerical errors due to the reduced float precision and range, the eigenvalue problem is split
            into multiple smaller solves. Preparing a new shift-invert LU decomposition is rather fast. """
        # Number of additional Lanczos construction vectors. Use a little more
        # than default recommendation to improve numerical stability and
        # precision:
        add_lzc_vec = 1.5 * self.neig
        n_lzc_vec = self.neig + add_lzc_vec  # Full vector space
        mul_thresh = mat.shape[0] * self.neig / self.gemm_dim_threshold
        if self.dtype == np.complex64 and mul_thresh > 1:
            n_lzc_vec = max(int(n_lzc_vec / mul_thresh), 40)
            # smaller neig subset, but always request a minimum set size
            neig = max(int(self.neig / mul_thresh), 10)
        else:
            neig = int(self.neig)
            n_lzc_vec = int(n_lzc_vec)
        target = self.targetval
        try:
            eival, eivec = self._solve(mat, target, neig, n_lzc_vec)
        except (ArithmeticError, TimeoutError) as err:
            # TODO: Consider flagging this as high load task and reschedule.
            # TaskManager should make sure only limited number of high load
            # tasks is active at once (not yet implemented)
            if self.verbose and self.dtype == np.complex64:
                stderr.write(str(err) + "\nFalling back to double precision solver (init).\n")
            eival, eivec = self._fallback(mat, err)
        ddp = DiagDataPoint(0, eival, eivec).sort_by_eival(inplace=True)
        prev_target = np.array([target, target])  # holds next targets in down/up direction, initial value
        range_factor = np.array([1.1, 1.1])
        prefactor = np.array([-1, 1])
        allowed_no_overlap = [False, False]
        for _ in range(int(5 * mul_thresh) + 20):  # this will only execute with single precision, because
            if not ddp.neig < self.neig:  # we abort this loop, as soon as enough eigenvalues are found
                break
            # extend search to both ends: From previous target value,
            # set new target to lie beyond the maximum distance to newly found eigenvalues
            # This should also cover cases where only values on one side of the original target are found (rare case)
            next_target = np.abs(np.array([ddp.eival.min(), ddp.eival.max()])-prev_target) * prefactor * range_factor + prev_target
            nt = np.argmin(np.abs(next_target-self.targetval))  # only step in the direction that is closer to targetval
            prev_neig = ddp.neig
            try:
                eival, eivec = self._solve(mat, next_target[nt], neig, n_lzc_vec)
            except (ArithmeticError, TimeoutError) as err:
                if self.verbose:
                    stderr.write(str(err) + "\nFalling back to double precision solver (%d found).\n" % prev_neig)
                eival, eivec = self._fallback(mat, err)
                ddp = DiagDataPoint(0, eival, eivec).sort_by_eival(inplace=True)
                break  # return with double precision solve
                # continue  # No convergence, we just retry
            t0 = rtime()
            # Problem with DDP.extend_by: Accuracy of shift-invert is lower, the
            # further away from target. With single precision, this can be quite
            # bad, and we are not able to clearly identify single eigenvalues
            # just by value and remove overlapping duplicates.
            # Solution: Check min/max values for overlap. Then shift sorted
            # eivals against each other to find best total eigenvalue match
            # (similar to bandalign algorithm, reusable?). Then we calculate the
            # mean from both sets of overlapping eigenvalues, weighted by each
            # point's distance to the target values.
            try:
                temp_ddp = ddp.stitch_with(0, eival, eivec, prev_target[nt], next_target[nt])
                n_new_eival = temp_ddp.neig - prev_neig  # With overlap, duplicates will be removed
                overlap = len(eival) - n_new_eival
                if overlap < 5 and not allowed_no_overlap[nt]:
                    raise NoOverlapError("Overlap of solutions too small (%d)." % (len(eival) - n_new_eival))
            except NoOverlapError as err:
                if allowed_no_overlap[nt]:
                    temp_ddp = ddp.subset(range(ddp.neig))  # get copy
                    temp_ddp.extend_by(0, eival, eivec).sort_by_eival(inplace=True)
                else:  # No overlap detected, reduce stepping for next iteration
                    # TODO: With improved stitching (results present on both
                    # sides), we'd be able to use also this result. With the
                    # current implementation, we have to throw it away.
                    range_factor[nt] = max(range_factor[nt] - 0.1, 0.95)
                    if self.verbose:
                        stderr.write("    " + str(err) + '\n')
                    continue
            except SubsetError as err:  # New solution is subset of previous.
                # Just continue as normal. Next target value will be further away.
                if self.verbose:
                    stderr.write("    " + str(err) + ". Target was %.3g.\n" % next_target[nt])
                    stderr.write("    Prev: %d from %.3g to %.3g, New: %d from %.3g to %.3g.\n" %
                                 (ddp.neig, ddp.eival.min(), ddp.eival.max(), len(eival), eival.min(), eival.max()))
                    stderr.write("    Range factor: %.3f\n" % range_factor[nt])
                # We know that there will be no states in the current solution
                # interval beyond the current borders. Therefore, we reset the
                # prev_target value to increase the step size in the next
                # iteration, but only if this is the first of each series of
                # subset errors (overlap not allowed):
                prev_target[nt] = next_target[nt] if allowed_no_overlap else eival.min() if nt == 1 else eival.max()
                range_factor[nt] = 0.95  # May not be larger then 1, otherwise it'd be possible to miss states.
                allowed_no_overlap[nt] = True  # Next step in this direction is very likely to have no overlap
                continue
            except ValueError as err:
                # Stitching can not return valid solution. Most likely cause is
                # that the overlap contains only fully degenerate solutions.
                # We could try to increase the overlap by adjusting the target
                # value or the number of requested eigenstates, but this issue
                # does not occur often, so it's easier and safer to fall back to
                # double precision directly.
                if self.verbose:
                    stderr.write(str(err) + "\nFalling back to double precision solver (%d found).\n" % prev_neig)
                eival, eivec = self._fallback(mat, err)
                ddp = DiagDataPoint(0, eival, eivec).sort_by_eival(inplace=True)
                break  # return with double precision solve
            if n_new_eival < 0.85 * len(eival) and overlap > 10:
                range_factor[nt] += 0.2
            if self.verbose:
                stderr.write("    %d new eigenstates. Stitched in %.3g s\n" % (n_new_eival, rtime()-t0))
                stderr.write("    Prev: %d from %.3g to %.3g, New: %d from %.3g to %.3g.\n" %
                             (ddp.neig, ddp.eival.min(), ddp.eival.max(), len(eival), eival.min(), eival.max()))
                stderr.write("    Range factor: %.3f\n" % range_factor[nt])
            prev_target[nt] = next_target[nt]
            ddp = temp_ddp
        if self.verbose and ddp.neig < self.neig:
            stderr.write("FOUND ONLY %d eivecs.\n" % ddp.neig)
        return ddp.eival, ddp.eivec

    def _fallback(self, mat: MatrixT, error: Exception) -> Any:
        """Step by step fallback routine:
        - Is called if single precision solver fails to converge.
        - Tries the double precision version for the full sized problem.
        - Should this also fail, create a (stable) CPU solver instance
            and solve this problem on CPU only"""
        try:
            if self.dtype == np.complex64:
                # Try double precision GPU solver
                eival, eivec = self._solve(mat, self.targetval, self.neig, int(2 * self.neig), dtype=np.complex128)
            else:
                # Reraise and catch the error to fall back to CPU solver
                raise error
        except TimeoutError as err:
            if self.verbose:
                stderr.write("Double precision GPU solver failed: " + str(err) + "\nFalling back to CPU solver.\n")
            cpu_solver = CustomShiftInvEigshSolver(self.num_processes, self.num_threads,
                                                   self.neig, self.targetval, self.shift_invert_solver)
            eival, eivec = cpu_solver.solve(mat)
        return eival, eivec

    def _solve(self, mat: MatrixT, target: float, neig: int, n_lzc_vec: int, dtype: np.dtype | None = None) -> Any:
        """Actual solve implementation: Configure shift-invert solver and use cupy eigsh."""
        shift_mat = mat - target * eye(mat.shape[0])
        shift_mat = shift_mat.astype(self.dtype if dtype is None else dtype)
        shift_invert_factorization = CupyFastSpLUInv(shift_mat, solver=self.shift_invert_solver, neig=n_lzc_vec)
        eival, eivec = cupy_eigsh(shift_invert_factorization, int(neig), ncv=int(n_lzc_vec))
        eival = (1.0 / eival.get().astype(np.double)) + target  # revert shift-invert of eigenvalues
        return eival, eivec.get().astype(np.complex128)

class JaxEighSolver(CustomShiftInvEigshSolver):
    def __init__(self, *args, **kwds) -> None:
        super().__init__(*args, **kwds)

        self.gpu_select: int = 0
        # Do not treat SIGCHLD as 'terminate', as this would interfere with
        # regular operation on jax.
        self.handle_sigchld: bool = False

    """Implements the JAX eigh solver for spare matrices."""
    def solve(self, mat: MatrixT) -> SolveResultT:
        try:
            mat = mat.todense()
        except AttributeError:  # Likely already a spare matrix
            pass

        # Distribute load across all available GPUs using
        # a Round Robin approach.
        gpu_select = self.gpu_select
        self.gpu_select += 1
        if self.gpu_select >= len(jax.devices("gpu")):
            self.gpu_select = 0

        with jax.default_device(jax.devices("gpu")[gpu_select]):
            eival, eivec = jnp.linalg.eigh(mat)

        # If self.neig is specified only return around self.targetval
        # This is for compatablity with the existing implementation.
        if self.neig is not None:
            target_index = jnp.searchsorted(eival, self.targetval)
            if target_index >= len(eival) - (self.neig // 2):
                # targetval either larger than all eivals or almost larger
                eival = eival[-self.neig:]
                eivec = eivec[:,-self.neig:]
            elif target_index - (self.neig // 2) <= 0:
                # targetval either smaller than all eivals or almost smaller
                eival = eival[:self.neig]
                eivec = eivec[:,:self.neig]
            else:
                # targetval somewhere in the middle of the array
                eival = eival[(target_index - self.neig // 2):(target_index + self.neig - self.neig // 2)]
                eivec = eivec[:,(target_index - self.neig // 2):(target_index + self.neig - self.neig // 2)]

        return np.array(eival), np.array(eivec)

class CupyFastSpLUInv(CupyLinOp if HAS_CUPY else LinearOperator):
    """
    SpLuInv:
       Helper class to repeatedly solve M*x = rhs
       using a sparse LU-decomposition of mat M.
       Do this on GPU using cupy CUDA library interface
       or on CPU with GPU<>CPU RAM transfers (faster in all tests so far).

       Note: If cupy package is not present, this class is redefined
       (but not used) as scipy based class to prevent import errors.
    """
    def __init__(self, mat: MatrixT, dtype: np.dtype | None = None, solver: str = 'cupy', neig: int = 0):
        super().__init__(mat.dtype if dtype is None else dtype, mat.shape)
        self.mat_dtype: np.dtype = mat.dtype
        t0 = rtime()
        if solver == 'cupy':
            # Factorize once on CPU (uses scipy default SuperLU),
            # solve multiple times on GPU
            cu_mat = cupy_csr_matrix(mat)
            self.M_lu_solve = cupy_factorized(cu_mat)
        elif solver == 'umfpack':
            # Factorize and solve on CPU (uses scipy default UMFPACK/SuperLU)
            self.M_lu_solve = factorized(mat)
        elif solver == 'superlu':
            # Chooses SuperLU directly.
            self.M_lu_solve = splu(mat).solve
        elif solver == 'pardiso':
            pardiso_invert = pyMKL.pardisoSolver(mat, mtype=-4)
            pardiso_invert.factor()
            self.M_lu_solve = pardiso_invert.solve
            self.solver_handle = pardiso_invert
        else:
            raise ValueError('Solver %s not supported' % solver)
        self.solver: str = solver
        self.use_cpu: bool = solver != 'cupy'
        self.n_mult: int = 0
        self.pref_time: float = rtime()
        self.total_t_lanczos: float = 0.0
        self.total_t_lu_solve: float = 0.0
        self.verbose: bool = cmdargs.sysargv.verbose
        self.neig: int = neig
        self.lanczos_iter: int = get_config_int('diag_solver_cupy_iterations', minval=1)
        if self.verbose:
            stderr.write("Initialized and factorized %s LU solver in %.3g s.\n" % (solver, rtime()-t0))

    def __del__(self) -> None:
        try:
            if self.solver == 'pardiso':
                self.solver_handle.clear()  # Clear memory of pardiso instance
            if self.verbose and self.n_mult > 0:
                if self.solver == 'cupy':
                    # As both parts run asynchronously on the GPU, we can not use CPU based
                    # timing here to distinguish both times. Just print the total time:
                    stderr.write('Total Lanczos + LU-Inv solve %.3g s. Used %d MatVec operations.\n'
                                 % (self.total_t_lanczos + self.total_t_lu_solve, self.n_mult))
                else:
                    stderr.write('Total Lanczos %.3g s (%.3g %%); Total LU-Inv solve %.3g s. Used %d MatVec operations.\n'
                                 % (self.total_t_lanczos, 100 * self.total_t_lanczos /
                                    (self.total_t_lu_solve + self.total_t_lanczos),
                                    self.total_t_lu_solve, self.n_mult))
        except AttributeError:
            # In some rare cases (e.g. memory allocation errors), this instance
            # will not have all (valid) attributes. This would throw additional
            # distracting secondary error messages, therefore, we suppress those
            # errors completely. The verbose output might also be suppressed, but
            # would not be of much use in this case anyway.
            pass

    def _matvec(self, rhs: MatrixT) -> MatrixT:
        """Custom definition of matrix-vector product M^-1 * rhs, where
        the matrix M^-1 would be the virtual inverse of the original M (not actually calculated)
        and the Vector rhs (usually an eigenvector candidate).
        """
        sleep(0)  # make this thread yield control to others, as we have to wait anyway
        if not cupy.all(cupy.isfinite(rhs)):
            raise ArithmeticError("Vector contains non-finite values. Range Overflow?")
        if self.use_cpu:
            rhs_cpu = cupy.asnumpy(rhs).astype(self.mat_dtype)
        else:
            rhs = rhs.astype(self.mat_dtype)
        t0 = rtime()
        t_lanczos = t0 - self.pref_time
        self.total_t_lanczos += t_lanczos
        t0 = rtime()
        self.n_mult += 1  # Count matvec multiplications = Lanzcos iterations
        x = self.M_lu_solve(rhs_cpu if self.use_cpu else rhs)
        if self.neig is not None and (self.n_mult - 5) // self.neig >= self.lanczos_iter:
            # stderr.write("Solver converges slowly! %d MatVecMult used. Aborting...\n" % self.n_mult)
            raise TimeoutError("Did not converge.")
        sleep(0)  # make this thread yield control to others, as we have to wait anyway
        # Extremely verbose debug output:
        # stderr.write('MatVecMul %d done (%.3f ms). %.3f ms between calls\n'
        #              % (self.n_mult, (rtime() - t0) * 1000, t_lanczos * 1000))
        self.total_t_lu_solve += rtime() - t0
        self.pref_time = rtime()
        if self.use_cpu:
            return cupy.asarray(x, dtype=self.dtype)
        return x.astype(self.dtype)

def solverconfig(num_cpus: int, modelopts: dict, script: str | None = None) -> DiagSolver:
    """Chooses a suitable sparse eigenvalue solver and sets its configuration.
    Returns a DiagSparseSolver instance.
    """
    global feast_not_found_warning, no_pardiso_warning, no_cupy_warning, cupy_warning_2d, no_jax_warning, no_slepc_warning
    if 'neig' not in modelopts:
        raise KeyError("Number of eigenvalues ('neig') not defined in modelopts")
    all_solvers = [
        'auto', 'automatic', 'feast', 'cupy_eigsh', 'jax_eigh', 'eigsh', 'superlu_eigsh',
        'umfpack_eigsh', 'pardiso_eigsh', 'slepc', 'slepcgpu'
    ]
    solver_config = get_config('diag_solver', choices = all_solvers).lower()
    if solver_config in ['auto', 'automatic']:
        if script in ['kdotpy-1d.py']:
            if HAS_PARDISO:
                solver_config = 'pardiso_eigsh'
            else:
                solver_config = 'eigsh'
                stderr.write("'diag_solver' hint: PARDISO can improve solution speed, but is not available. Consider installing packages 'MKL' and 'pyMKL' (see Wiki).\n")
            if HAS_CUPY:
                stderr.write("'diag_solver' hint: Solver 'cupy_eigsh' is available and could improve performance under certain conditions. Double precision recommended.\n")
        elif script in ['kdotpy-ll.py'] and HAS_CUPY and modelopts['ll_mode'] == 'full' and modelopts['neig'] >= 200:
            solver_config = 'eigsh'
            stderr.write("'diag_solver' hint: Solver 'cupy_eigsh' is available and could improve performance under certain conditions. Single precision recommended.\n")
        else:
            solver_config = 'eigsh'
        stderr.write(f"'diag_solver' configuration 'automatic': Choosing '{solver_config}'")  # to be continued
        stderr.write(".\n" if script is None else f" for script '{script}'.\n")

    worker_type = get_config('diag_solver_worker_type', choices = ['auto', 'automatic', 'process', 'thread']).lower()
    if worker_type in ['auto', 'automatic']:
        if 'cupy' in solver_config or 'jax' in solver_config or 'slepcgpu' in solver_config:
            worker_type = 'thread'
        else:
            worker_type = 'process'
        stderr.write("'diag_solver_worker_type' configuration 'automatic': Choosing '%s' for solver '%s'.\n" % (worker_type, solver_config))

    if 'll_mode' in modelopts and modelopts['ll_mode'] != 'full':
        # Renormalize total number of eigenvalues
        if int(np.ceil(modelopts['neig'] / modelopts['ll_max'])) < 6:
            stderr.write("Warning (lldiagonalization.hll): Requested number of eigenstates leads to < 6 eigenstates per LL index. Use minimum of 6 states per LL index instead. It is recommended to increase the value for neig and/or decrease the value for llmax or nll.\n")
        modelopts['neig'] = int(np.ceil(modelopts['neig'] / modelopts['ll_max']))

    num_threads = cmdargs.threads()

    if solver_config == 'feast':
        if not HAS_FEAST:
            if feast_not_found_warning:
                stderr.write("Warning (diagsolver): FEAST solver could not be loaded. Please make the Intel MKL available. Falling back to legacy solver (scipy.sparse.linalg.eigsh).\n")
                feast_not_found_warning = False  # only issue once
            solver_config = 'eigsh'
        else:
            if 'erange' not in modelopts:
                raise KeyError("Target energy range ('erange') not defined in modelopts")
            emin, emax = modelopts['erange']
            return FeastSolver(num_cpus, num_threads if num_threads is not None else 1,
                               modelopts['neig'], emin, emax, worker_type = worker_type)

    if 'eigs' in solver_config and 'energy' not in modelopts:
        raise KeyError("Target energy ('energy') not defined in modelopts")

    if solver_config == 'cupy_eigsh':
        if not HAS_CUPY:
            if no_cupy_warning:
                stderr.write("Warning (diagsolver): CUDA solver could not be loaded. Please check your CUPY package. Falling back to legacy CPU 'eigsh' solver (with UMFPACK/SuperLU).\n")
                no_cupy_warning = False  # only issue once
            solver_config = 'umfpack_eigsh'
            worker_type = 'process'
        else:
            # Set the float data type (precision) for calculations using the cupy solver.
            cupy_dtype = get_config('diag_solver_cupy_dtype', choices = ['single', 'double']).lower()
            if script == 'kdotpy-2d.py':
                if cupy_warning_2d:
                    stderr.write(f"Warning (diagsolver): Using the CUDA solver with {script} occasionally fails with eigenvalues at incorrect energies. Check your results and choose a different solver if you notice problems.\n")
                    cupy_warning_2d = False  # only issue once

            if not HAS_PARDISO:
                if no_pardiso_warning:
                    stderr.write("Warning (diagsolver): PARDISO solver could not be loaded. Please make the Intel MKL available. Falling back to legacy shift invert factorization solver (UMFPACK/SuperLU).\n")
                    no_pardiso_warning = False  # only issue once
                shift_invert_solver = 'umfpack'
            else:
                shift_invert_solver = 'pardiso'
            return CupyShiftInvEigshSolver(
                num_cpus, num_threads if num_threads is not None else 1,
                modelopts['neig'], modelopts['energy'], shift_invert_solver,
                dtype = cupy_dtype, worker_type = worker_type)

    if solver_config == 'jax_eigh':
        if not HAS_JAX:
            if no_jax_warning:
                stderr.write("Warning (diagsolver): JAX solver could not be loaded. Please check your JAX package. Falling back to legacy CPU 'eigsh' solver (with UMFPACK/SuperLU).\n")
                no_jax_warning = False  # only issue once
            solver_config = 'umfpack_eigsh'
            worker_type = 'process'
        else:
            return JaxEighSolver(num_cpus, num_threads if num_threads is not None else 1,
                                 modelopts['neig'], modelopts['energy'], None, worker_type = worker_type)

    if solver_config == 'pardiso_eigsh':
        if not HAS_PARDISO:
            if no_pardiso_warning:
                stderr.write("Warning (diagsolver): PARDISO solver could not be loaded. Please make the Intel MKL available. Falling back to legacy shift invert factorization solver (UMFPACK/SuperLU).\n")
                no_pardiso_warning = False  # only issue once
            solver_config = 'superlu_eigsh'
        else:
            return CustomShiftInvEigshSolver(num_cpus, num_threads if num_threads is not None else 1,
                                             modelopts['neig'], modelopts['energy'], 'pardiso',
                                             worker_type = worker_type)
    if solver_config == 'umfpack_eigsh':
        # Falls back to 'superlu_eigsh' silently, if UMFPACK is not available
        return CustomShiftInvEigshSolver(num_cpus, num_threads if num_threads is not None else 1,
                                         modelopts['neig'], modelopts['energy'], 'umfpack',
                                         worker_type = worker_type)

    if solver_config == 'superlu_eigsh':
        # This is essentially equal to 'eigsh', but enables us to get some more detailed timing information.
        return CustomShiftInvEigshSolver(num_cpus, num_threads if num_threads is not None else 1,
                                         modelopts['neig'], modelopts['energy'], 'superlu',
                                         worker_type = worker_type)

    if solver_config == 'slepc':
        if not HAS_SLEPC:
            if no_slepc_warning:
                stderr.write("Warning (diagsolver): SLEPC solver could not be loaded. Please check your SLEPC package. Falling back to legacy CPU 'eigsh'.\n")
                no_slepc_warning = False  # only issue once
            solver_config = 'eigsh'
            worker_type = 'process'
        else:
            return SLEPCSolver(num_cpus, num_threads if num_threads is not None else 1,
                            modelopts['neig'], modelopts['energy'],get_config('diag_solver_slepc_options'), worker_type = worker_type)

    if solver_config == 'slepcgpu':
        if not HAS_SLEPC:
            if no_slepc_warning:
                stderr.write("Warning (diagsolver): SLEPC solver could not be loaded. Please check your SLEPC package. Falling back to legacy CPU 'eigsh'.\n")
                no_slepc_warning = False  # only issue once
            solver_config = 'eigsh'
            worker_type = 'process'
        else:
            return SLEPCGPUSolver(get_config('diag_solver_slepc_options'), num_cpus, num_threads if num_threads is not None else 1,
                                 modelopts['neig'], modelopts['energy'], None, worker_type = worker_type)

    if solver_config == 'eigsh':
        if isinstance(modelopts['energy'], list):
            return EigshMultiSolver(num_cpus, num_threads if num_threads is not None else 1,
                                    modelopts['neig'], modelopts['energy'], worker_type = worker_type)
        else:
            return EigshSolver(num_cpus, num_threads if num_threads is not None else 1,
                               modelopts['neig'], modelopts['energy'], worker_type = worker_type)
    else:
        raise ValueError("Invalid value for variable solver_config")
