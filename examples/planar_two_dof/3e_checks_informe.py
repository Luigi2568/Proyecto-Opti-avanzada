# Extensión IEE3002 (Grupo 24): recalcula los números "de desarrollo" que aparecen en el documento de
# continuación (Secciones III, IV, VI y VII), para que todos salgan de la misma máquina.
#
# Uso (desde la raíz del repo, con el venv activado; toma unos minutos):
#   PYTHONPATH=. python examples/planar_two_dof/3e_checks_informe.py
import pathlib
import pickle
import sys

import cvxpy as cp
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from examples.planar_two_dof import P_EXAMPLE_2_DOF
from utils_aux import get_linear_ordered_state_constraints, get_linear_box_constraints, get_hyper_box_corners
from offline_pipeline_b2_lmi_controller import (
    default_solver, compile_lmi_problem, solve_base, solve_lmi_ccp, sample_model_error_terms, norms_P,
    _controller_from_EY, _s_t,
)

P_DATA = P_EXAMPLE_2_DOF / "data"
P_BASE = P_DATA / "dof_2_ef_0.2"


def titulo(t):
    print(f"\n=== {t}")


def main():
    solver = default_solver()
    print(f"solver de los SDP: {solver}")
    sys_ = np.load(P_BASE / "system.npz")
    A, B = sys_["A"], sys_["B"]
    cons = np.load(P_BASE / "data_constraints.npz")
    u_amp, v_amp = float(cons["u_amp"]), float(cons["v_amp"])
    verts = np.load(P_BASE / "data_model_error_verts.npz")
    wc_x = B @ verts["verts_acc"] + verts["verts_state_disc"]
    verts_w = get_hyper_box_corners(4, value=1) * wc_x
    A_x, _ = get_linear_ordered_state_constraints(2, p_amp=np.pi, v_amp=v_amp)
    A_u, _ = get_linear_box_constraints(2, amp=u_amp)
    weights = (1 / 0.1, 1 / v_amp, 1 / u_amp, (A_x.shape[0] + A_u.shape[0]) / 2)
    with (P_BASE / "data_controllers.pckl").open("rb") as fp:
        res_aut = pickle.load(fp)

    # ------------------------------------------------------------------------------------------
    titulo("III-A. SDP acoplado (4x4) frente al desacoplado de los autores (20 valores de rho)")
    err_E, err_Y, err_loss, offdiag = [], [], [], []
    for (_, E_a, Y_a, _, _, _, rho, loss_a) in res_aut:
        b = solve_base(A, B, A_x, A_u, verts_w, rho, weights, solver)
        err_E.append(np.abs(b["E"] - E_a).max() / np.abs(E_a).max())
        err_Y.append(np.abs(b["Y"] - Y_a).max() / np.abs(Y_a).max())
        err_loss.append(abs(b["loss"] - loss_a) / loss_a)
        offdiag.append(max(abs(b["E"][0, 1]), abs(b["E"][0, 3]), abs(b["E"][1, 2]), abs(b["E"][2, 3])))
    print(f"error relativo máx: E {max(err_E):.1e} | Y {max(err_Y):.1e} | costo {max(err_loss):.1e} | "
          f"término fuera de la diagonal máx {max(offdiag):.1e}")

    titulo("III-A. Clarabel SIN escalamiento, rho = 0.80 (la LMI de contracción queda violada)")
    rho = res_aut[0][6]
    prob, var = compile_lmi_problem(A, B, A_x, A_u, verts_w, None, None, rho, weights, mode="base")
    prob.solve(solver=cp.CLARABEL)
    E, Y = var["E"].value, var["Y"].value
    mineig = np.linalg.eigvalsh(np.block([[rho ** 2 * E, (A @ E + B @ Y).T], [A @ E + B @ Y, E]])).min()
    print(f"estado {prob.status} | costo {prob.value:.4f} (óptimo de los autores {res_aut[0][7]:.4f}) | "
          f"autovalor mínimo de la LMI de contracción {mineig:.1e}")

    # ------------------------------------------------------------------------------------------
    rng = np.random.default_rng(0)   # mismas muestras que usa 1b_run_offline_lmi.py con seed = 0
    from examples.planar_two_dof import P_EXAMPLE_2_DOF as _P
    from problem_scenario import ProblemScenarioMassAllPin
    ps = ProblemScenarioMassAllPin(str(_P / "manipulator" / "two_link.urdf"), P_BASE, np.r_[20, 20], u_amp_nom=20,
                                   v_amp_nom=2, dt=1e-2, error_frac=0.2, ignore_gravity=True)
    pool_a, pool_v = sample_model_error_terms(ps, B, v_amp, nr_theta=300, nr_q=20, rng=rng)

    def base_init(rho):
        b = solve_base(A, B, A_x, A_u, verts_w, rho, weights, solver)
        P, K, Ps, Pis = _controller_from_EY(b["E"], b["Y"])
        s, t = _s_t(K, Pis)
        ka, kv = norms_P(Ps, pool_a).max(), norms_P(Ps, pool_v).max()
        return b, dict(P_sqrt=Ps, lam_a=s / ka, lam_v=t / kv), rho + ka * s + kv * t

    titulo("III-C. Chequeo de consistencia: (24) con rho_tilde_max = 0.999, rho = 0.90")
    b, init, _ = base_init(0.9)
    r = solve_lmi_ccp(A, B, A_x, A_u, verts_w, pool_a, pool_v, 0.9, weights, solver, mode="tightening",
                      rho_tilde_max=0.999, init=init, T=b["T"])
    relE = np.abs(r["sol"]["E"] - b["E"]).max() / np.abs(b["E"]).max()
    print(f"costo (24) {r['hist'][-1]['obj']:.5f} | costo base {b['loss']:.5f} | error relativo en E {relE:.1e}")

    titulo("IV. Fase 1: mínimo rho_tilde alcanzable con (24), partiendo de tres valores de lambda")
    for rho in (0.8, 0.9):
        b, init, rt_base = base_init(rho)
        mins = []
        for f in (1.0, 0.3, 3.0):
            ini = dict(init, lam_a=init["lam_a"] * f, lam_v=init["lam_v"] * f)
            r = solve_lmi_ccp(A, B, A_x, A_u, verts_w, pool_a, pool_v, rho, weights, solver, mode="rho_tilde",
                              init=ini, T=b["T"], max_iter=60)
            mins.append(r["hist"][-1]["rho_tilde_pool"])
        print(f"rho {rho:.2f}: base {rt_base:.4f} | mínimo con (24) desde lambda x1, x0.3, x3: "
              + ", ".join(f"{m:.4f}" for m in mins))

    # ------------------------------------------------------------------------------------------
    titulo("VI-C. Método base: mínimo rho_tilde en la grilla de rho (de los .pckl de esta máquina)")
    for ef in ("0.3", "0.35", "0.4"):
        p = P_DATA / f"dof_2_ef_{ef}_lmi_results.pckl"
        if not p.exists():
            print(f"{ef}: no está {p.name}")
            continue
        with p.open("rb") as fp:
            rr = pickle.load(fp)["results"]
        base = sorted((x["rho"], x["ev"]["rho_tilde"]) for x in rr if x["method"] == "base")
        i_min = int(np.argmin([t for _, t in base]))
        print(f"{ef}: mínimo {base[i_min][1]:.4f} en rho = {base[i_min][0]:.2f} | "
              f"rho = {base[0][0]:.2f}: {base[0][1]:.4f} | rho = {base[-1][0]:.2f}: {base[-1][1]:.4f}")

    titulo("VI-B y VI-C. Comparación corredor a corredor, LMI robusto contra base (results.csv de esta máquina)")
    for ef in ("0.2", "0.3", "0.35"):
        try:
            rb = pd.read_csv(P_DATA / f"dof_2_ef_{ef}_base" / "results.csv")
            rl = pd.read_csv(P_DATA / f"dof_2_ef_{ef}_lmi_robust" / "results.csv")
        except FileNotFoundError:
            print(f"{ef}: faltan results.csv")
            continue
        rb, rl = rb[rb.name == "ft"].set_index("run_nr"), rl[rl.name == "ft"].set_index("run_nr")
        runs = rb.index.intersection(rl.index)
        d = (rl.loc[runs, "t2g"] - rb.loc[runs, "t2g"]) / 100
        print(f"{ef}: {len(runs)} corredores | LMI gana {int((d < 0).sum())}, empata {int((d == 0).sum())}, "
              f"pierde {int((d > 0).sum())} | diferencia media {d.mean():+.3f} s "
              f"({d.mean() / (rb.loc[runs, 't2g'].mean() / 100) * 100:+.1f} %)")

    # ------------------------------------------------------------------------------------------
    titulo("VII. Bug c_un_sq y L sin recalcular (archivos guardados de los autores)")
    rt = dict(np.load(P_BASE / "rtube_controller.npz"))
    K, E = rt["K"], rt["E"]
    true = np.array([a @ K @ E @ K.T @ a for a in np.vstack([np.eye(2), -np.eye(2)])])
    delta = np.sqrt(rt["w_bar_sq"]) / (1 - rt["rho"])
    print(f"c_u_sq guardado {np.ravel(rt['c_u_sq'])} | correcto {true}")
    print(f"tubo rígido delta = {float(delta):.2f} | ajuste de aceleración negativa: guardado "
          f"{np.sqrt(np.ravel(rt['c_u_sq'])[2]) * delta:.3f}, correcto {np.sqrt(true[2]) * delta:.2f} rad/s^2")
    ft = dict(np.load(P_BASE / "ftube_controller.npz"))
    s = np.linalg.norm(ft["K"] @ ft["P_inv_sqrt"], 2)
    t = np.linalg.norm(ft["P_inv_sqrt"][2:, :], 2)
    print(f"L guardado {float(ft['L']):.6f} -> rho_tilde {float(ft['rho'] + ft['L']):.5f} | "
          f"L con las kappa convergidas {float(ft['a'] * s + ft['b'] * t):.6f} -> "
          f"rho_tilde {float(ft['rho'] + ft['a'] * s + ft['b'] * t):.5f} | kappa_0 de los autores {float(ft['c']):.5f}")


if __name__ == "__main__":
    main()
