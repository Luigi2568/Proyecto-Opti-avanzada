# Extensión IEE3002 (Grupo 24): verificación densa de las constantes de un controlador guardado.
#
# En este robot planar M(q) y C(q, qd) dependen solo de q2, y la incertidumbre son las masas de los
# dos eslabones, así que el peor caso de kappa_a, kappa_v y de la contracción robusta es un máximo
# sobre (q2, m1, m2): se puede recorrer con una grilla densa en vez de muestras al azar.
# (kappa_v: el máximo sobre la caja de velocidades está en sus vértices, por linealidad en qd).
#
# Uso:  python examples/planar_two_dof/1c_verify_dense_grid.py dof_2_ef_0.2_lmi_robust [--contraction robust]
import argparse
import pathlib
import pickle
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from examples.planar_two_dof import P_EXAMPLE_2_DOF
from problem_scenario import ProblemScenarioMassAllPin
from utils_aux import get_hyper_box_corners
from offline_pipeline_b2_lmi_controller import norms_P, robust_contraction


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dir_name")
    parser.add_argument("--contraction", choices=["product", "robust"], default=None,
                        help="cómo se calculó rho_tilde = rho + L al guardar (por defecto se infiere)")
    parser.add_argument("--n-q2", type=int, default=721)
    parser.add_argument("--n-m", type=int, default=21)
    args = parser.parse_args()

    p_dir = P_EXAMPLE_2_DOF / "data" / args.dir_name
    with (p_dir / "ps.pckl").open("rb") as fp:
        ef = pickle.load(fp)["error_frac"]
    ps = ProblemScenarioMassAllPin(str(P_EXAMPLE_2_DOF / "manipulator" / "two_link.urdf"), p_dir, np.r_[20, 20],
                                   u_amp_nom=20, v_amp_nom=2, dt=1e-2, error_frac=ef, ignore_gravity=True)
    ctrl = dict(np.load(p_dir / "ftube_controller.npz"))
    sys_ = np.load(p_dir / "system.npz")
    A, B = sys_["A"], sys_["B"]
    v_amp = float(np.load(p_dir / "data_constraints.npz")["v_amp"])
    P_sqrt, P_inv_sqrt, K = ctrl["P_sqrt"], ctrl["P_inv_sqrt"], ctrl["K"]

    dyn_nom = ps.get_nominal_dynamics()
    q2s = np.linspace(-np.pi, np.pi, args.n_q2)
    ms = 1 + np.linspace(-ef, ef, args.n_m)
    qd_verts = get_hyper_box_corners(2, value=v_amp)
    G_a, G_v = [], []
    for r1 in ms:
        for r2 in ms:
            dyn_err = ps.get_err_dyn_from_ratios(np.r_[1.0, r1, r2])   # la masa 0 (base fija) no influye
            for q2 in q2s:
                q = np.r_[0.0, q2]
                M_nom = dyn_nom.mass_matrix(q)
                M_err = dyn_err.mass_matrix(q)
                M_inv = np.linalg.inv(M_nom + M_err)
                G_a.append(B @ (M_inv @ M_err))
                for qd in qd_verts:
                    G_v.append(B @ (M_inv @ dyn_err.coroli_matrix(q, qd)))
    G_a, G_v = np.array(G_a), np.array(G_v)

    a_grid = norms_P(P_sqrt, G_a).max()
    b_grid = norms_P(P_sqrt, G_v).max()
    rho_r_grid = robust_contraction(P_sqrt, P_inv_sqrt, A, B, K, G_a).max()
    s = np.linalg.norm(K @ P_inv_sqrt, ord=2)
    t = np.linalg.norm(P_inv_sqrt[2:, :], ord=2)
    rt_saved = float(ctrl["rho"] + ctrl["L"])
    contraction = args.contraction or ("robust" if abs(ctrl["L"] - ctrl["b"] * t) < 1e-9 else "product")
    if contraction == "robust":
        rt_grid = rho_r_grid + b_grid * t
    else:
        rt_grid = float(ctrl["rho"]) + a_grid * s + b_grid * t
    print(f"{args.dir_name}  (error_frac {ef}, grilla {args.n_m}x{args.n_m} masas x {args.n_q2} q2 = {G_a.shape[0]} puntos)")
    print(f"  kappa_a: guardado {float(ctrl['a']):.6f} | grilla {a_grid:.6f}")
    print(f"  kappa_v: guardado {float(ctrl['b']):.6f} | grilla {b_grid:.6f}")
    print(f"  rho_tilde ({contraction}): guardado {rt_saved:.5f} | grilla {rt_grid:.5f}"
          f"   [contracción robusta en la grilla: {rho_r_grid:.5f}]")


if __name__ == "__main__":
    main()
