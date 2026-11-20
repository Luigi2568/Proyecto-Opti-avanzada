# Extensión IEE3002 (Grupo 24): métricas de seguridad de las trayectorias simuladas, a partir de motion/*.pckl.
#
#   - Distancia mínima a los obstáculos:
#       * en el espacio de configuraciones: la SCDF (cspace_data.npz) en cada estado real, en rad;
#       * en el espacio de trabajo: distancia de los eslabones (segmentos) a los discos, en mm.
#     Si alguna es negativa, hubo colisión.
#   - Margen de torque: tau = M0(q) a + C0(q, qd) qd + g0(q), el mismo torque que usan los autores para
#     certificar los límites (verify_torque_limits), con a la aceleración que aplica el controlador
#     auxiliar. Se evalúa al inicio y al final de cada paso y se compara con el límite de 20 N·m.
#
# Uso (desde la raíz del repo):
#   PYTHONPATH=. python examples/planar_two_dof/3f_safety_metrics.py dof_2_ef_0.2 dof_2_ef_0.2_lmi_robust
#   PYTHONPATH=. python examples/planar_two_dof/3f_safety_metrics.py dof_2_ef_0.2 dof_2_ef_0.2_rt_fix --method rt
import argparse
import pathlib
import pickle
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from examples.planar_two_dof import P_EXAMPLE_2_DOF
from examples.planar_two_dof.world import DiskWorld
from examples.planar_two_dof.manipulator.man_two_link import distance_line_segment_and_point
from problem_scenario import ProblemScenarioMassAllPin


def workspace_distance(world, q):
    """Distancia mínima (m) entre los eslabones (segmentos) y los obstáculos (discos); < 0 = colisión."""
    pts = world.man.get_wireframe(q)
    return min(
        distance_line_segment_and_point(l_s, l_e, c) - r
        for c, r in zip(world.obs_p, world.obs_r)
        for l_s, l_e in zip(pts[:-1], pts[1:])
    )


def corridor_metrics(steps, world, dyn_nom, tau_lims):
    """Recorre un corredor: estados reales (x_t y los pasos del controlador auxiliar) y aceleraciones aplicadas."""
    n = dyn_nom.config_dim
    states, taus = [], []
    for step in steps:
        # el tubo rígido (TubeResults) no guarda x_t ni la aceleración aplicada: ahí solo hay distancias
        x_prev = getattr(step, "x_t", None)
        if x_prev is not None:
            states.append(np.asarray(x_prev))
        for row in np.atleast_2d(step.states_aux_controller):
            x_next = row[:2 * n]
            if row.size > 2 * n and x_prev is not None:
                a = row[2 * n:]
                for x in (np.asarray(x_prev), x_next):   # el torque cambia durante el paso: inicio y final
                    q, qd = x[:n], x[n:]
                    taus.append(dyn_nom.mass_matrix(q) @ a + dyn_nom.coroli(q, qd) + dyn_nom.gravity(q))
            states.append(x_next)
            x_prev = x_next
    states = np.array(states)
    qs = states[:, :n]
    d_cspace = np.asarray(world.sdf(qs)).min()
    d_work = min(workspace_distance(world, q) for q in qs)
    if not taus:
        return d_cspace, d_work, np.nan, 0, 0
    taus = np.abs(np.array(taus))
    return d_cspace, d_work, taus.max(), (taus > np.asarray(tau_lims)[None, :] + 1e-9).any(axis=1).sum(), len(taus)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dir_names", nargs="+")
    parser.add_argument("--method", default="ft", help="ft, rt, nom o nom_star")
    parser.add_argument("--n-a", type=int, default=1, help="lee motion_na{n_a}/ si n_a > 1")
    args = parser.parse_args()
    suffix = "" if args.n_a == 1 else f"_na{args.n_a}"

    world = DiskWorld.from_example()
    ps = ProblemScenarioMassAllPin(str(P_EXAMPLE_2_DOF / "manipulator" / "two_link.urdf"), P_EXAMPLE_2_DOF,
                                   np.r_[20, 20], u_amp_nom=20, v_amp_nom=2, dt=1e-2, error_frac=0.2,
                                   ignore_gravity=True)
    dyn_nom, tau_lims = ps.get_nominal_dynamics(), np.asarray(ps.torque_limits, dtype=float)

    print(f"{'carpeta':30s} {'corr.':>5s} | {'dist. C-espacio [rad]':>21s} | {'dist. espacio trabajo [mm]':>26s} | "
          f"{'|tau| máx [N·m]':>14s} {'margen mín':>10s} {'sobre límite':>12s}")
    print(f"{'':30s} {'':>5s} | {'mín':>9s} {'media mín':>11s} | {'mín':>11s} {'media mín':>14s} |")
    for name in args.dir_names:
        files = sorted((P_EXAMPLE_2_DOF / "data" / name / f"motion{suffix}").glob(f"{args.method}~*.pckl"))
        if not files:
            print(f"{name:30s} sin archivos {args.method}~*.pckl en motion{suffix}/")
            continue
        rows = []
        for f in files:
            with f.open("rb") as fp:
                rows.append(corridor_metrics(pickle.load(fp), world, dyn_nom, tau_lims))
        d_c, d_w, t_max, n_over, n_tot = map(np.array, zip(*rows))
        dist = f"{name:30s} {len(files):5d} | {d_c.min():9.4f} {d_c.mean():11.4f} | {d_w.min() * 1e3:11.1f} " \
               f"{d_w.mean() * 1e3:14.1f} |"
        if np.isnan(t_max).all():
            print(f"{dist} {'(no se guarda la aceleración)':>38s}")
        else:
            margin = (tau_lims.min() - np.nanmax(t_max)) / tau_lims.min() * 100
            print(f"{dist} {np.nanmax(t_max):14.2f} {margin:9.1f}% {int(n_over.sum()):6d}/{int(n_tot.sum())}")


if __name__ == "__main__":
    main()
