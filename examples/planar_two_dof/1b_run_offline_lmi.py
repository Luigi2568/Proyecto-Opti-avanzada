# Extensión IEE3002 (Grupo 24): etapa offline del diseño LMI para el ejemplo planar de 2 GDL.
#
# Uso (desde la raíz del repo, igual que los otros scripts del ejemplo):
#   python examples/planar_two_dof/1b_run_offline_lmi.py --error-frac 0.2
#   python examples/planar_two_dof/1b_run_offline_lmi.py --error-frac 0.3 --methods base lmi_robust
#
# Si existe data/dof_2_ef_{ef}/ con la etapa "a" de los autores (system, data_constraints,
# data_model_error_verts), se usan esos datos. Si no, se calculan aquí con muestreo rápido.
#
# Salidas en data/:
#   dof_2_ef_{ef}_lmi_results.pckl   todos los controladores de la grilla y su evaluación
#   dof_2_ef_{ef}_{método}/          carpeta lista para 2_run_all.py con ftube_controller.npz del método
import argparse
import logging
import pathlib
import pickle
import shutil
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from examples.planar_two_dof import P_EXAMPLE_2_DOF
from problem_scenario import ProblemScenarioMassAllPin
from utils_aux import get_linear_double_integrator_discrete_dynamics, get_hyper_box_corners
from offline_pipeline_a_disturbances import verify_torque_limits
from offline_pipeline_b2_lmi_controller import (
    run_lmi_controller_pipeline,
    sample_acc_error_box,
    sample_disc_and_gravity_errors,
    pick_best,
    ftube_npz_dict,
    save_results,
    summary_table,
)

P_DATA = P_EXAMPLE_2_DOF / "data"
P_URDF = P_EXAMPLE_2_DOF / "manipulator" / "two_link.urdf"


def get_problem(error_frac, p_cached_dir):
    return ProblemScenarioMassAllPin(
        str(P_URDF).replace("\\", "/"),
        p_cached_dir,
        np.r_[20, 20],
        u_amp_nom=20,
        v_amp_nom=2,
        dt=1e-2,
        error_frac=error_frac,
        ignore_gravity=True
    )


def load_or_compute_stage_a(ps, p_base, logger, rng):
    """Datos de la etapa 'a' (A, B, u_amp, v_amp y caja W). Usa los de los autores si existen."""
    files = [p_base / f for f in ("system.npz", "data_constraints.npz", "data_model_error_verts.npz")]
    if all(f.exists() for f in files):
        logger.debug(f"usando etapa a de {p_base}")
        sys_, cons, verts = [np.load(f) for f in files]
        A, B = sys_["A"], sys_["B"]
        u_amp, v_amp = float(cons["u_amp"]), float(cons["v_amp"])
        verts_acc, verts_state_disc = verts["verts_acc"], verts["verts_state_disc"]
        source = "autores"
    else:
        logger.debug(f"no hay etapa a en {p_base}: se calcula con muestreo rápido")
        A, B = get_linear_double_integrator_discrete_dynamics(ps.config_dim, dt=ps.dt, method="zoh")
        A, B = np.array(A), np.array(B)
        u_amp, v_amp = verify_torque_limits(ps, logger=logger, nr_samples_q=10_000)
        u_amp, v_amp = float(u_amp), float(v_amp)
        verts_acc = sample_acc_error_box(ps, u_amp, v_amp, nr_theta=500, nr_q=40, rng=rng)
        disc = sample_disc_and_gravity_errors(ps, A, B, u_amp, v_amp, nr_samples=20_000, rng=rng)
        verts_state_disc = np.abs(disc).max(axis=0)   # con gravedad ignorada, B g_tilde = 0
        source = "muestreo rápido"
    wc_x = B @ verts_acc + verts_state_disc
    verts_w = get_hyper_box_corners(B.shape[0], value=1) * wc_x
    stage_a = dict(A=A, B=B, u_amp=u_amp, v_amp=v_amp, verts_acc=verts_acc, verts_state_disc=verts_state_disc)
    return stage_a, verts_w, source


def write_run_dir(p_dir, ps, stage_a, ftube, p_base):
    """Carpeta con el mismo formato que dof_2_ef_*/ para correr 2_run_all.py con este controlador."""
    p_dir.mkdir(parents=True, exist_ok=True)
    np.savez(p_dir / "system.npz", A=stage_a["A"], B=stage_a["B"])
    np.savez(p_dir / "data_constraints.npz", u_amp=stage_a["u_amp"], v_amp=stage_a["v_amp"])
    np.savez(p_dir / "data_model_error_verts.npz", verts_acc=stage_a["verts_acc"],
             verts_state_disc=stage_a["verts_state_disc"])
    np.savez(p_dir / "ftube_controller.npz", **ftube)
    if (p_base / "rtube_controller.npz").exists():
        shutil.copy(p_base / "rtube_controller.npz", p_dir / "rtube_controller.npz")
    # ps.pckl solo con tipos de Python (sin pathlib ni numpy): se abre en Linux y Windows, y con numpy 1 o 2
    with (p_dir / "ps.pckl").open("wb") as fp:
        pickle.dump(dict(
            path_urdf=str(P_URDF).replace("\\", "/"), p_cached_dir=str(p_dir),
            torque_limits=[float(x) for x in ps.torque_limits], v_amp_nom=float(ps.v_amp_nom),
            u_amp_nom=float(ps.u_amp_nom), dt=float(ps.dt), error_frac=float(ps.frac),
            ignore_gravity=bool(ps.ignore_gravity)
        ), fp)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--error-frac", type=float, default=0.2)
    parser.add_argument("--methods", nargs="+",
                        default=["base", "base_robust_analysis", "lmi_product", "lmi_robust"])
    parser.add_argument("--save", nargs="+", default=["base", "base_robust_analysis", "lmi_robust"],
                        help="métodos cuyo controlador elegido se guarda en una carpeta para 2_run_all.py")
    parser.add_argument("--nr-rhos", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    ef = args.error_frac
    name = f"dof_2_ef_{ef}"
    p_base = P_DATA / name
    logger = logging.getLogger("lmi")
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(handler)

    rng = np.random.default_rng(args.seed)
    ps = get_problem(ef, p_base)
    stage_a, verts_w, source = load_or_compute_stage_a(ps, p_base, logger, rng)
    logger.debug(f"error_frac {ef} | etapa a: {source} | u_amp {stage_a['u_amp']} | W: {verts_w.max(axis=0)}")

    results, samples = run_lmi_controller_pipeline(
        ps, stage_a["A"], stage_a["B"], stage_a["u_amp"], stage_a["v_amp"], verts_w, logger,
        rhos=np.linspace(0.8, 0.99, args.nr_rhos), methods=args.methods, seed=args.seed
    )
    save_results(P_DATA / f"{name}_lmi_results.pckl", results, samples)
    print(summary_table(results, methods=args.methods))

    for m in args.save:
        best = pick_best(results, m)
        if best is None:
            logger.debug(f"{m}: sin controlador con rho_tilde < 1, no se guarda carpeta")
            continue
        p_dir = P_DATA / f"{name}_{m}"
        write_run_dir(p_dir, ps, stage_a, ftube_npz_dict(best["ev"], best["sol"]), p_base)
        logger.debug(f"{m}: guardado en {p_dir}")


if __name__ == "__main__":
    main()
