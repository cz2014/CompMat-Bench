# Workflows

[Back to the benchmark](../README.md) · [Download](../docs/download.md) ·
[Software environment](../docs/evaluation.md#agent-software-environment)

Each workflow asks an agent to carry out a sequence of research calculations
and report final quantities. **The agent carries out every step itself,
including the simulations in most workflows.** The
paper evaluates C01–C08 with full and reduced methodological guidance; C09 and
C10 are supplementary packages without agent evaluation.

## Workflow catalogue

| Workflow | Title | Evaluated in the paper |
|---|---|---|
| [C01](C01/) | WS2/WSe2 moire: relaxation, piezoelectric charge and electron potential | yes |
| [C02](C02/) | Se-pair substitutions: space groups, tight-binding models and spin splittings | yes |
| [C03](C03/) | BAs: force constants, isotope-disordered dielectric function and near-field transmission | yes |
| [C04](C04/) | Si/Ge interface: force constants, phonon reflection and specularity | yes |
| [C05](C05/) | Zn(1-x)Mg(x)O: ordered cells, polarization paths and Landau free energy | yes |
| [C06](C06/) | CrN: magnetic ground state, Wannier exchange constants and spin waves | yes |
| [C07](C07/) | V2CoAl orderings: relaxation, magnetic states and exchange couplings | yes |
| [C08](C08/) | Ni island: EAM saddle, machine-learned refinement and curvature certification | yes |
| [C09](C09/) | Curie temperature and compression response in CuInP2S6 | no, supplementary |
| [C10](C10/) | Molecular reaction energies and chain propagation | no, supplementary |

The [manifest](manifest.json) records these titles and evaluation status.
Supplementary workflows have additional documentation in [C09](C09/README.md)
and [C10](C10/README.md).

## Run and grade a workflow

Download and unpack the [core release package](../docs/download.md), which
includes the staged workflow inputs. From the benchmark root, select
`workflows/Cxx/agent/LG/` for full guidance or `workflows/Cxx/agent/LA/` for
reduced guidance, and copy that package into a fresh working directory.

Give your agent the selected package and its prompt, with the required software
and offline manuals available. Keep grading references, other packages and
trajectories outside its accessible environment. The reported workflow
evaluation used **one 128-core node and five hours per workflow**, with no
internet access.

The agent should write the final quantities in the answer format specified by
the prompt. Grade the answer outside the agent's environment, from the benchmark
root. For example, for C01:

```sh
python3 workflows/grader.py workflows/C01/grader/reference.json /path/to/answer.json
```

See the [software table](../docs/evaluation.md#agent-software-environment) and
[runtime records](runtime/) for the environment used in the paper.

## Licensed inputs

C06, C07 and C10 run VASP. POTCAR files are not redistributed. Each affected
package's `inputs/potcars/POTCARS_NOT_INCLUDED.md` lists the files to supply from
your own licensed installation before running the workflow.

Some manuals also require separate access; see the
[reference-library exclusions](../reflib/README.md#not-included).
