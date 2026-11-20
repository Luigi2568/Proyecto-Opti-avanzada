# Extensión IEE3002 (Grupo 24): tiempos de cómputo del MPC en línea, a partir de los motion/*.pckl.
#
# Cada paso guarda times = (cómputo completo del paso, asignación de bolas, prob.solve() de cvxpy,
# tiempo que reporta el solver). Los tres primeros son tiempo de reloj (time.time()). El último lo informa
# Clarabel y puede ser MAYOR que el de reloj (en la Raspberry cuenta tiempo de CPU de varios hilos), así que
# para el límite de tiempo real hay que mirar el cómputo completo: debe ser < n_a * Ts = 10 ms (n_a = 1).
#
# Uso (desde la raíz del repo):
#   PYTHONPATH=. python examples/planar_two_dof/3c_solve_times.py dof_2_ef_0.2 dof_2_ef_0.2_lmi_robust
#   PYTHONPATH=. python examples/planar_two_dof/3c_solve_times.py dof_2_ef_0.2 --method rt
import argparse
import pathlib
import pickle
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from examples.planar_two_dof import P_EXAMPLE_2_DOF

LIMIT_MS = 10.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dir_names", nargs="+")
    parser.add_argument("--method", default="ft", help="ft, rt, nom o nom_star")
    parser.add_argument("--n-a", type=int, default=1, help="lee motion_na{n_a}/ si n_a > 1; el límite es n_a * 10 ms")
    args = parser.parse_args()
    suffix = "" if args.n_a == 1 else f"_na{args.n_a}"
    limit_ms = LIMIT_MS * args.n_a

    print(f"{'carpeta':30s} {'pasos':>6s} | {'cómputo del paso: media':>23s} {'p95':>6s} {'máx':>6s} {f'<{limit_ms:.0f}ms':>6s} | "
          f"{'cvxpy solve':>11s} | {'reportado solver':>16s}")
    for name in args.dir_names:
        files = sorted((P_EXAMPLE_2_DOF / "data" / name / f"motion{suffix}").glob(f"{args.method}~*.pckl"))
        if not files:
            print(f"{name:30s} sin archivos {args.method}~*.pckl en motion{suffix}/")
            continue
        times = []
        for f in files:
            with f.open("rb") as fp:
                times.extend(step.times for step in pickle.load(fp))
        t = np.array(times) * 1e3   # ms
        total, solve, solver = t[:, 0], t[:, 2], t[:, 3]
        print(f"{name:30s} {len(t):6d} | {total.mean():20.2f} ms {np.percentile(total, 95):6.2f} "
              f"{total.max():6.2f} {np.mean(total < limit_ms) * 100:5.1f}% | {solve.mean():8.2f} ms | "
              f"{solver.mean():13.2f} ms")


if __name__ == "__main__":
    main()
