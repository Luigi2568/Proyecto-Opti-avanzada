# Extensión IEE3002 (Grupo 24) sobre el código de Wullt et al. (ABB, 2025).
#
# Diseño LMI conjunto del controlador auxiliar (P, K) y de la contracción del tubo flexible.
#
# Método base (offline_pipeline_b_controller.py):
#   1) se resuelve el SDP (11) para una grilla de rho (diseña P, K para el tubo RÍGIDO),
#   2) DESPUÉS se estiman kappa_a, kappa_v, kappa_0 por muestreo y
#   3) recién ahí se revisa que rho_tilde = rho + kappa_a*s + kappa_v*t < 1,
#      con s = ||K P^{-1/2}|| y t = ||V P^{-1/2}||.
#
# Aquí kappa_a^2, kappa_v^2, s^2 y t^2 son variables del SDP (LMIs (24a)-(24c) del informe, vía
# complemento de Schur, afines en (sigma, E, Y)) y se impone (24d):
#       rho + kappa_a*s + kappa_v*t <= rho_tilde_max.
# El producto kappa*s = sqrt(sigma_kappa * sigma_s) se acota por
#       kappa*s <= 1/2 (lam*sigma_kappa + sigma_s/lam),   lam > 0,
# que es LINEAL en las variables al cuadrado y exacta si lam = s/kappa. Se itera actualizando lam
# (procedimiento convexo-cóncavo): cada iteración es factible y el objetivo no aumenta.
#
# Las LMIs (24a) solo se pueden imponer en un conjunto finito de muestras (theta_l, q_l, qd_l):
# se parte con pocas y se agregan las más violadas de un pool grande (planos de corte).
#
# Variante "robust" (la que funciona, ver README_IEE3002.md): en vez de acotar kappa_a*s, se impone
# directamente la contracción del lazo cerrado con error de modelo, ||A + BK - B M_tilde_l K||_P <= rho,
# que es una LMI en (E, Y) para rho fijo. Así rho_tilde = rho + kappa_v*t y no hace falta el CCP.
#
# Notación del código: a = kappa_a, b = kappa_v, c = kappa_0 (igual que los autores).

import pickle
import time

import cvxpy as cp
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import sqrtm

from utils_aux import (
    compute_model_error,
    get_hyper_box_corners,
    get_hyper_box_corners_from_limits,
    get_linear_box_constraints,
    get_linear_ordered_state_constraints,
    compute_projected_matrix,
)


def default_solver():
    return cp.MOSEK if "MOSEK" in cp.installed_solvers() else cp.CLARABEL


# ----------------------------------------------------------------------------------------------
# 1. Muestras del error de modelo
# ----------------------------------------------------------------------------------------------

def _position_limits(problem):
    if problem.position_limits is None:
        n = problem.config_dim
        return np.c_[-np.ones(n), np.ones(n)].T * np.pi
    return problem.position_limits


def sample_model_error_terms(problem, B, v_amp, nr_theta=200, nr_q=20, include_vertices=True, rng=None):
    """
    Muestras de las matrices que definen kappa_a y kappa_v (ec. (13a)-(13b)):
        G_a = B M_tilde(q),        M_tilde = M^{-1} M_err
        G_v = B C_tilde(q, qd),    C_tilde = M^{-1} C_err(q, qd)
    kappa_a = max ||P^{1/2} G_a||, kappa_v = max ||P^{1/2} G_v||.
    Como C_err es lineal en qd y la norma es convexa, el máximo sobre la caja de velocidades está
    en sus vértices: por eso G_v se evalúa solo en los vértices de qd.
    Las masas (theta) se muestrean en los vértices de la caja de incertidumbre + al azar.
    Retorna G_a (N_a, m_x, m_p), G_v (N_v, m_x, m_p).
    """
    rng = np.random.default_rng() if rng is None else rng
    dyn_nom = problem.get_nominal_dynamics()
    n = problem.config_dim
    lims = _position_limits(problem)
    ratios_rand = 1 + rng.uniform(-problem.frac, problem.frac, size=(nr_theta, problem.parameter_dim))
    if include_vertices:
        ratios_wc = 1 + get_hyper_box_corners(problem.parameter_dim, value=problem.frac)
        ratios = np.vstack([ratios_wc, ratios_rand])
    else:
        ratios = ratios_rand
    qd_verts = get_hyper_box_corners(n, value=v_amp)
    conf_cube = get_hyper_box_corners_from_limits(lims.T)
    G_a, G_v = [], []
    for ratio in ratios:
        dyn_err = problem.get_err_dyn_from_ratios(ratio)
        Qs = np.vstack([rng.uniform(lims[0], lims[1], size=(nr_q, n)), conf_cube])
        for q in Qs:
            M_nom = dyn_nom.mass_matrix(q)
            M_err = dyn_err.mass_matrix(q)
            M_inv = np.linalg.inv(M_nom + M_err)
            G_a.append(B @ (M_inv @ M_err))
            for qd in qd_verts:
                G_v.append(B @ (M_inv @ dyn_err.coroli_matrix(q, qd)))
    return np.array(G_a), np.array(G_v)


def sample_acc_error_box(problem, u_amp, v_amp, nr_theta=200, nr_q=20, rng=None):
    """
    Caja (por componente) que contiene el error de aceleración Delta_theta (ec. (6)),
    equivalente rápida de compute_bound_and_disturbance_set (pipeline a). Solo se usa para
    niveles de incertidumbre en que no se corrió el pipeline a de los autores.
    """
    rng = np.random.default_rng() if rng is None else rng
    dyn_nom = problem.get_nominal_dynamics()
    n = problem.config_dim
    lims = _position_limits(problem)
    ratios = np.vstack([
        1 + get_hyper_box_corners(problem.parameter_dim, value=problem.frac),
        1 + rng.uniform(-problem.frac, problem.frac, size=(nr_theta, problem.parameter_dim))
    ])
    u_verts = get_hyper_box_corners(n, value=u_amp)
    qd_verts = get_hyper_box_corners(n, value=v_amp)
    conf_cube = get_hyper_box_corners_from_limits(lims.T)
    verts = np.zeros(n)
    for ratio in ratios:
        dyn_err = problem.get_err_dyn_from_ratios(ratio)
        Qs = np.vstack([rng.uniform(lims[0], lims[1], size=(nr_q, n)), conf_cube])
        for q in Qs:
            M_nom = dyn_nom.mass_matrix(q)
            M_err = dyn_err.mass_matrix(q)
            M_inv = np.linalg.inv(M_nom + M_err)
            M_tilde = M_inv @ M_err
            g_tilde = M_inv @ dyn_err.gravity(q)
            for qd in qd_verts:
                c_tilde_qd = M_inv @ dyn_err.coroli(q, qd)
                deltas = -(u_verts @ M_tilde.T) - c_tilde_qd - g_tilde
                verts = np.maximum(verts, np.abs(deltas).max(axis=0))
    return verts


def sample_disc_and_gravity_errors(problem, A, B, u_amp, v_amp, nr_samples=2000, rng=None):
    """
    Muestras del vector  B g_tilde + Delta_disc  que define kappa_0 (ec. (13c)), con el mismo
    procedimiento de los autores (integración continua vs. modelo discreto en un paso Ts).
    No dependen de (P, K), así que se calculan una vez y sirven para evaluar cualquier controlador.
    """
    rng = np.random.default_rng() if rng is None else rng
    dyn_nom = problem.get_nominal_dynamics()
    n = problem.config_dim
    lims = _position_limits(problem)
    dt = problem.dt
    out = []
    ratios_wc = 1 + get_hyper_box_corners(problem.parameter_dim, value=problem.frac)
    for i in range(nr_samples):
        # la mitad de las muestras en vértices (masas, velocidad y aceleración extremas), donde suele
        # estar el peor caso; la otra mitad al azar, como los autores
        if i % 2 == 0:
            dyn_err = problem.get_err_dyn_from_ratios(ratios_wc[rng.integers(ratios_wc.shape[0])])
            qd = rng.choice([-1.0, 1.0], size=n) * v_amp
            acc = rng.choice([-1.0, 1.0], size=n) * u_amp
        else:
            dyn_err = problem.get_err_dyn_random()
            qd = rng.uniform(-v_amp, v_amp, size=n)
            acc = rng.uniform(-u_amp, u_amp, size=n)
        q = rng.uniform(lims[0], lims[1], size=n)
        x = np.r_[q, qd]

        def x_dot_fl(t, x_t, q_dd_t):
            q_t, q_d_t = np.split(x_t, 2)
            return np.r_[q_d_t, q_dd_t + compute_model_error(dyn_nom, dyn_err, q_t, q_d_t, q_dd_t)]

        x_ode = solve_ivp(x_dot_fl, [0, dt], x, args=(acc,)).y[:, -1]
        delta_theta = compute_model_error(dyn_nom, dyn_err, q, qd, acc)
        disc_err = x_ode - (A @ x + B @ (acc + delta_theta))
        M_inv = np.linalg.inv(dyn_nom.mass_matrix(q) + dyn_err.mass_matrix(q))
        g_tilde = M_inv @ dyn_err.gravity(q)
        out.append(B @ g_tilde + disc_err)
    return np.array(out)


def norms_P(P_sqrt, G):
    """||P^{1/2} G_l|| (norma espectral) para todas las muestras G (N, m_x, k) o vectores (N, m_x)."""
    if G.ndim == 2:
        return np.linalg.norm(G @ P_sqrt.T, axis=1)
    return np.linalg.norm(P_sqrt @ G, ord=2, axis=(1, 2))


# ----------------------------------------------------------------------------------------------
# 2. SDP
# ----------------------------------------------------------------------------------------------

def compile_lmi_problem(
        A, B, A_x, A_u, verts_w, G_a, G_v, rho, weights,
        rho_tilde_max=None, lam_a=1.0, lam_v=1.0, mode="tightening", eps_E=1e-8, T=None
):
    """
    mode = "base":        SDP (11) del método base (sin las nuevas LMIs) en su versión acoplada.
    mode = "tightening":  mismo objetivo (11a) + LMIs (24a)-(24c) + (24d) linealizada <= rho_tilde_max.
    mode = "rho_tilde":   minimiza la cota linealizada de rho_tilde (fase 1: cuánto se puede contraer).
    mode = "robust":      objetivo (11a) + contracción robusta ||A + BK - G_l K||_P <= rho en las muestras G_a.

    T: escalamiento diagonal del estado, x = T z (solo numérico). En las coordenadas originales E tiene
    entradas entre ~1e-5 y ~1e-2 y el solver termina "optimal_inaccurate"; con T = sqrt(diag(E)) de una
    solución previa, E_z queda ~ O(1). Todas las LMIs son congruencias, así que el problema es el mismo:
    E = T E_z T,  Y = Y_z T,  y las sigma (kappa^2, s^2, t^2) no cambian.
    """
    m_x, m_u = B.shape
    m_p = m_x // 2
    n_x = A_x.shape[0]
    n_p = n_x // 2
    n_u = A_u.shape[0]
    c_p_wht, c_v_wht, c_u_wht, w_wht = weights
    T = np.ones(m_x) if T is None else np.asarray(T)
    T_inv = 1 / T
    A = T_inv[:, None] * A * T[None, :]
    B = T_inv[:, None] * B
    A_x = A_x * T[None, :]
    verts_w = verts_w * T_inv[None, :]
    G_a = None if G_a is None else G_a * T_inv[None, :, None]
    G_v = None if G_v is None else G_v * T_inv[None, :, None]

    E = cp.Variable((m_x, m_x), PSD=True, name="E")
    Y = cp.Variable((m_u, m_x), name="Y")
    c_x_sq = cp.Variable((n_x,), nonneg=True, name="c_x_sq")
    c_u_sq = cp.Variable((n_u,), nonneg=True, name="c_u_sq")
    w_bar_sq = cp.Variable(nonneg=True, name="w_bar_sq")

    tightening = 1 / (2 * (1 - rho)) * (
        w_wht * w_bar_sq
        + c_p_wht * cp.sum(c_x_sq[:n_p])
        + c_v_wht * cp.sum(c_x_sq[n_p:])
        + c_u_wht * cp.sum(c_u_sq)
    )
    const = [
        E >> np.eye(m_x) * eps_E,
        # (11b) contracción
        cp.bmat([[(rho ** 2) * E, (A @ E + B @ Y).T], [A @ E + B @ Y, E]]) >> 0,
    ]
    # (11c)-(11e) ajuste de restricciones y perturbación de peor caso (igual que el método base)
    for i in range(n_x):
        const += [cp.bmat([[c_x_sq[i][None, None], A_x[i, None] @ E], [(A_x[i, None] @ E).T, E]]) >> 0]
    for j in range(n_u):
        const += [cp.bmat([[c_u_sq[j][None, None], A_u[j, None] @ Y], [(A_u[j, None] @ Y).T, E]]) >> 0]
    for v in verts_w:
        const += [cp.bmat([[w_bar_sq[None, None], v[None]], [v[:, None], E]]) >> 0]

    variables = dict(E=E, Y=Y, c_x_sq=c_x_sq, c_u_sq=c_u_sq, w_bar_sq=w_bar_sq)
    if mode == "base":
        return cp.Problem(cp.Minimize(tightening), const), variables
    if mode == "robust":
        # Variante: contracción robusta. El error de modelo entra como -B M_tilde(q) (a_bar + K e), así que
        #   e+ = (A + B K - B M_tilde K) e + [-B M_tilde a_bar - B C_tilde qd - B g_tilde + D_disc]
        # y basta  ||A + B K - B M_tilde_l K||_P <= rho  para todas las muestras l:
        #   [[rho^2 E, (A E + B Y - G_l Y)^T], [A E + B Y - G_l Y, E]] >= 0,   G_l = B M_tilde_l.
        # Con eso rho_tilde = rho + kappa_v * t (el término kappa_a * s desaparece de la contracción).
        for G in G_a:
            cl = A @ E + B @ Y - G @ Y
            const += [cp.bmat([[(rho ** 2) * E, cl.T], [cl, E]]) >> 0]
        return cp.Problem(cp.Minimize(tightening), const), variables

    sig_a = cp.Variable(nonneg=True, name="sig_a")   # kappa_a^2
    sig_v = cp.Variable(nonneg=True, name="sig_v")   # kappa_v^2
    sig_s = cp.Variable(nonneg=True, name="sig_s")   # s^2 = ||K P^{-1/2}||^2
    sig_t = cp.Variable(nonneg=True, name="sig_t")   # t^2 = ||V P^{-1/2}||^2
    I_p = np.eye(m_p)
    # (24a): ||P^{1/2} G|| <= kappa  <=>  G^T E^{-1} G <= kappa^2 I  <=>  [[sig I, G^T], [G, E]] >= 0
    for G in G_a:
        const += [cp.bmat([[sig_a * I_p, G.T], [G, E]]) >> 0]
    for G in G_v:
        const += [cp.bmat([[sig_v * I_p, G.T], [G, E]]) >> 0]
    # (24b): ||K P^{-1/2}|| = ||Y E^{-1/2}|| <= s  <=>  [[sig_s I, Y], [Y^T, E]] >= 0
    const += [cp.bmat([[sig_s * np.eye(m_u), Y], [Y.T, E]]) >> 0]
    # (24c): ||V P^{-1/2}||^2 = ||V E V^T|| <= t^2,  V = [0, I]   (en coordenadas z: V T E_z T V^T)
    T_v = np.diag(T[m_p:])
    const += [T_v @ E[m_p:, m_p:] @ T_v << sig_t * I_p]
    # (24d) linealizada: kappa*s <= 1/2 (lam*kappa^2 + s^2/lam)
    rho_tilde_lin = rho + 0.5 * (lam_a * sig_a + sig_s / lam_a) + 0.5 * (lam_v * sig_v + sig_t / lam_v)

    if mode == "tightening":
        if rho_tilde_max is not None:
            const += [rho_tilde_lin <= rho_tilde_max]
        objective = tightening
    elif mode == "rho_tilde":
        # pequeño peso en el objetivo base para que la solución no quede mal escalada
        objective = rho_tilde_lin + 1e-6 * tightening
    else:
        raise ValueError(mode)
    variables.update(sig_a=sig_a, sig_v=sig_v, sig_s=sig_s, sig_t=sig_t)
    return cp.Problem(cp.Minimize(objective), const), variables


def _solve(prob, solver):
    try:
        prob.solve(solver=solver, verbose=False)
    except cp.error.SolverError:
        return False
    return prob.status in ("optimal", "optimal_inaccurate")


def _values(var, T):
    """Valores de las variables en coordenadas originales (deshace x = T z)."""
    val = {k: v.value for k, v in var.items()}
    val["E"] = T[:, None] * val["E"] * T[None, :]
    val["Y"] = val["Y"] * T[None, :]
    return val


def scaling_from_E(E):
    return np.sqrt(np.diag(E))


def _controller_from_EY(E, Y):
    E = 0.5 * (E + E.T)
    P = np.linalg.inv(E)
    P = 0.5 * (P + P.T)
    K = Y @ P
    return P, K, np.real(sqrtm(P)), np.real(sqrtm(E))


def _s_t(K, P_inv_sqrt):
    m_x = P_inv_sqrt.shape[0]
    m_p = m_x // 2
    return np.linalg.norm(K @ P_inv_sqrt, ord=2), np.linalg.norm(P_inv_sqrt[m_p:, :], ord=2)


def solve_base(A, B, A_x, A_u, verts_w, rho, weights, solver, T=None):
    """SDP (11) del método base. Si no se da T, se resuelve una vez sin escalar para obtenerlo."""
    if T is None:
        prob, var = compile_lmi_problem(A, B, A_x, A_u, verts_w, None, None, rho, weights, mode="base")
        if not _solve(prob, solver):
            return None
        T = scaling_from_E(var["E"].value)
    prob, var = compile_lmi_problem(A, B, A_x, A_u, verts_w, None, None, rho, weights, mode="base", T=T)
    if not _solve(prob, solver):
        return None
    val = _values(var, T)
    return dict(E=val["E"], Y=val["Y"], c_x_sq=val["c_x_sq"], c_u_sq=val["c_u_sq"], w_bar_sq=val["w_bar_sq"],
                loss=prob.value, rho=rho, T=T, status=prob.status)


def robust_contraction(P_sqrt, P_inv_sqrt, A, B, K, G_a):
    """Tasa de contracción con error de modelo: ||A + B K - G_l K||_P para cada muestra G_l = B M_tilde_l."""
    A_cl = A + B @ K - G_a @ K
    return np.linalg.norm(P_sqrt @ A_cl @ P_inv_sqrt, ord=2, axis=(1, 2))


def solve_robust(
        A, B, A_x, A_u, verts_w, pool_a, rho, weights, solver, T=None, init_P_sqrt=None,
        n_init=20, n_add=10, viol_tol=1e-4, max_iter=30, logger=None
):
    """
    SDP de la variante robusta: objetivo (11a) + (11c)-(11e) + contracción robusta en las muestras
    activas de B M_tilde (planos de corte sobre el pool). Un SDP convexo por iteración; no requiere CCP.
    """
    T = np.ones(B.shape[0]) if T is None else T
    act = {int(np.argmin(np.linalg.norm(pool_a, axis=(1, 2))))}   # la muestra más cercana a M_tilde = 0
    if init_P_sqrt is not None:
        act |= set(np.argsort(-norms_P(init_P_sqrt, pool_a))[:n_init].tolist())
    time_s = time.time()
    sol, n_cuts = None, 0
    for it in range(max_iter):
        prob, var = compile_lmi_problem(A, B, A_x, A_u, verts_w, pool_a[sorted(act)], None, rho, weights,
                                        mode="robust", T=T)
        if not _solve(prob, solver):
            return dict(status=prob.status, rho=rho, sol=sol, n_cuts=n_cuts)
        val = _values(var, T)
        P, K, P_sqrt, P_inv_sqrt = _controller_from_EY(val["E"], val["Y"])
        rc = robust_contraction(P_sqrt, P_inv_sqrt, A, B, K, pool_a)
        sol = dict(E=val["E"], Y=val["Y"], P=P, K=K, P_sqrt=P_sqrt, P_inv_sqrt=P_inv_sqrt,
                   c_x_sq=val["c_x_sq"], c_u_sq=val["c_u_sq"], w_bar_sq=val["w_bar_sq"], loss=prob.value,
                   rho_r_pool=rc.max(), n_act=len(act), status=prob.status)
        new = [i for i in np.argsort(-rc) if rc[i] > rho * (1 + viol_tol) and i not in act][:n_add]
        if not new:
            break
        act |= set(new)
        n_cuts += 1
    if logger is not None:
        logger.debug(f"robust rho {rho:0.3f} | iters {it + 1} cuts {n_cuts} activas {len(act)} "
                     f"rho_r(pool) {sol['rho_r_pool']:0.4f} loss {sol['loss']:0.4f} time {time.time() - time_s:0.1f}s")
    return dict(status="optimal", rho=rho, sol=sol, n_cuts=n_cuts)


def solve_lmi_ccp(
        A, B, A_x, A_u, verts_w, pool_a, pool_v, rho, weights, solver,
        mode="tightening", rho_tilde_max=None, init=None, T=None,
        n_init=20, n_add=10, viol_tol=1e-3, max_iter=40, tol=1e-5, logger=None
):
    """
    Procedimiento convexo-cóncavo + planos de corte.
    init: dict con P_sqrt (para elegir las primeras muestras activas) y lam_a, lam_v.
          Si se parte de un punto factible para (24d) linealizada, todas las iteraciones son factibles.
    T: escalamiento numérico (ver compile_lmi_problem).
    """
    T = np.ones(B.shape[0]) if T is None else T
    P_sqrt0 = init["P_sqrt"]
    lam_a, lam_v = init["lam_a"], init["lam_v"]
    act_a = set(np.argsort(-norms_P(P_sqrt0, pool_a))[:n_init].tolist())
    act_v = set(np.argsort(-norms_P(P_sqrt0, pool_v))[:n_init].tolist())
    act_a |= set(init.get("act_a", set()))
    act_v |= set(init.get("act_v", set()))
    hist = []
    sol = None
    n_cuts = 0
    time_s = time.time()
    for it in range(max_iter):
        prob, var = compile_lmi_problem(
            A, B, A_x, A_u, verts_w, pool_a[sorted(act_a)], pool_v[sorted(act_v)], rho, weights,
            rho_tilde_max=rho_tilde_max, lam_a=lam_a, lam_v=lam_v, mode=mode, T=T
        )
        if not _solve(prob, solver):
            return dict(status=prob.status, rho=rho, rho_tilde_max=rho_tilde_max, hist=hist, sol=sol)
        val = _values(var, T)
        P, K, P_sqrt, P_inv_sqrt = _controller_from_EY(val["E"], val["Y"])
        # planos de corte: ¿alguna muestra del pool viola (24a) más de viol_tol?
        na, nv = norms_P(P_sqrt, pool_a), norms_P(P_sqrt, pool_v)
        ka, kv = np.sqrt(val["sig_a"]), np.sqrt(val["sig_v"])
        new_a = [i for i in np.argsort(-na) if na[i] > ka * (1 + viol_tol) and i not in act_a][:n_add]
        new_v = [i for i in np.argsort(-nv) if nv[i] > kv * (1 + viol_tol) and i not in act_v][:n_add]
        s, t = _s_t(K, P_inv_sqrt)
        rho_tilde_exact = rho + na.max() * s + nv.max() * t
        hist.append(dict(it=it, obj=prob.value, status=prob.status, lam_a=lam_a, lam_v=lam_v,
                         rho_tilde_pool=rho_tilde_exact, n_act_a=len(act_a), n_act_v=len(act_v),
                         cut=bool(new_a or new_v)))
        sol = dict(E=val["E"], Y=val["Y"], P=P, K=K, P_sqrt=P_sqrt, P_inv_sqrt=P_inv_sqrt,
                   c_x_sq=val["c_x_sq"], c_u_sq=val["c_u_sq"], w_bar_sq=val["w_bar_sq"],
                   sig_a=val["sig_a"], sig_v=val["sig_v"], sig_s=val["sig_s"], sig_t=val["sig_t"],
                   act_a=set(act_a), act_v=set(act_v), lam_a=lam_a, lam_v=lam_v)
        if new_a or new_v:
            act_a |= set(new_a)
            act_v |= set(new_v)
            n_cuts += 1
            continue
        # actualización convexo-cóncava: lam = s/kappa hace exacta la cota en el punto actual
        lam_a_new = np.sqrt(val["sig_s"] / max(val["sig_a"], 1e-16))
        lam_v_new = np.sqrt(val["sig_t"] / max(val["sig_v"], 1e-16))
        converged = (
            len(hist) > 1 and not hist[-2]["cut"]
            and abs(hist[-2]["obj"] - hist[-1]["obj"]) <= tol * max(1.0, abs(hist[-1]["obj"]))
            and abs(lam_a_new / lam_a - 1) < 1e-3 and abs(lam_v_new / lam_v - 1) < 1e-3
        )
        lam_a, lam_v = lam_a_new, lam_v_new
        if converged:
            break
    if logger is not None:
        logger.debug(
            f"rho {rho:0.3f} mode {mode} rt_max {rho_tilde_max} | iters {len(hist)} cuts {n_cuts} "
            f"obj {hist[-1]['obj']:0.5f} rho_tilde(pool) {hist[-1]['rho_tilde_pool']:0.4f} "
            f"time {time.time() - time_s:0.1f}s"
        )
    return dict(status="optimal", rho=rho, rho_tilde_max=rho_tilde_max, hist=hist, sol=sol)


# ----------------------------------------------------------------------------------------------
# 3. Evaluación común (mismo estimador para el método base y el propuesto)
# ----------------------------------------------------------------------------------------------

def evaluate_controller(E, Y, rho, verif_a, verif_v, verif_c, A_x, A_u, u_amp, v_amp,
                        A=None, B=None, contraction="product"):
    """
    Constantes a, b, c, contracción rho_tilde, tubo en reposo delta_f, r_p y el 'fitness' de los
    autores (pick_ftube_controller_from_results), todo con el mismo conjunto de muestras.
    contraction = "product": rho_tilde = rho + a*s + b*t            (paper, ec. (15))
    contraction = "robust":  rho_tilde = max_l ||A + BK - G_l K||_P + b*t   (variante robusta)
    Para guardar en el formato de los autores (el MPC usa rho + L) se reporta rho_eff y L con
    rho_eff + L = rho_tilde.
    """
    P, K, P_sqrt, P_inv_sqrt = _controller_from_EY(E, Y)
    m_x = P.shape[0]
    m_p = m_x // 2
    a = norms_P(P_sqrt, verif_a).max()
    b = norms_P(P_sqrt, verif_v).max()
    c = norms_P(P_sqrt, verif_c).max()
    s, t = _s_t(K, P_inv_sqrt)
    rho_tilde_product = rho + a * s + b * t
    rho_r = robust_contraction(P_sqrt, P_inv_sqrt, A, B, K, verif_a).max() if A is not None else np.nan
    rho_tilde_robust = rho_r + b * t
    if contraction == "product":
        rho_eff, L = rho, a * s + b * t
    elif contraction == "robust":
        rho_eff, L = rho_r, b * t
    else:
        raise ValueError(contraction)
    rho_tilde = rho_eff + L
    c_x = np.linalg.norm(P_inv_sqrt.T @ A_x.T, axis=0)
    c_u = np.linalg.norm((K @ P_inv_sqrt).T @ A_u.T, axis=0)
    P_q = compute_projected_matrix(P, m_p)
    r_p = np.sqrt(1 / np.linalg.eigvalsh(P_q)).max()
    beta_unit = a * np.sqrt(m_p) + b * np.sqrt(m_p) + c
    if rho_tilde < 1:
        s_ = beta_unit / (1 - rho_tilde)
        delta_f = c / (1 - rho_tilde)
    else:
        s_ = delta_f = np.inf
    fitness = max(s_ * c_x[:m_x].max() / 0.1, s_ * c_x[m_x:].max() / v_amp, s_ * c_u.max() / u_amp)
    return dict(
        rho=rho, rho_eff=rho_eff, L=L, rho_tilde=rho_tilde, contraction=contraction,
        rho_tilde_product=rho_tilde_product, rho_tilde_robust=rho_tilde_robust, rho_r=rho_r,
        a=a, b=b, c=c, s=s, t=t, delta_f=delta_f, r_p=r_p,
        delta_f_rad=r_p * delta_f, fitness=fitness, P=P, K=K, E=E, Y=Y, P_sqrt=P_sqrt, P_inv_sqrt=P_inv_sqrt,
        c_x=c_x, c_u=c_u
    )


def ftube_npz_dict(ev, sol):
    """Mismo formato que ftube_controller.npz del método base (lo lee FlexibleTubeNaiveCorridorController)."""
    return dict(
        rho=ev["rho_eff"], L=ev["L"], P=ev["P"], K=ev["K"], E=ev["E"], Y=ev["Y"],
        P_sqrt=ev["P_sqrt"], P_inv_sqrt=ev["P_inv_sqrt"],
        c_x_sq=sol["c_x_sq"], c_u_sq=sol["c_u_sq"], w_bar_sq=sol["w_bar_sq"],
        a=ev["a"], b=ev["b"], c=ev["c"]
    )


# ----------------------------------------------------------------------------------------------
# 4. Pipeline completo para un nivel de incertidumbre
# ----------------------------------------------------------------------------------------------

def run_lmi_controller_pipeline(
        problem, A, B, u_amp, v_amp, verts_w, logger,
        rhos=np.linspace(0.8, 0.99, 20),
        rho_tilde_grid=(0.90, 0.92, 0.94, 0.96, 0.98, 0.99, 0.995),
        pool_size=(300, 20), verif_size=(1000, 50), nr_disc=20000,
        methods=("base", "base_robust_analysis", "lmi_product", "lmi_robust"),
        solver=None, seed=0
):
    """
    Para cada rho de la grilla calcula los controladores de cada método y los evalúa con el MISMO
    conjunto de verificación (independiente del pool usado para diseñar):
      base                  SDP (11) de los autores; rho_tilde = rho + a s + b t          (ec. (15))
      base_robust_analysis  mismo (P, K) que base; rho_tilde con la cota de contracción robusta
                            (separa cuánto viene del análisis y cuánto del diseño)
      lmi_product           formulación (24) del informe: SDP (11) + (24a)-(24d) con CCP, para cada
                            rho_tilde_max de la grilla (más la fase 1: mínimo rho_tilde alcanzable)
      lmi_robust            variante: SDP (11) con la contracción robusta en las muestras de B M_tilde
    Selección final (pick_best) igual a los autores: menor 'fitness' entre los con rho_tilde < 1.
    """
    solver = solver or default_solver()
    rng = np.random.default_rng(seed)
    m_x, m_u = B.shape
    m_p = m_x // 2
    A_x, _ = get_linear_ordered_state_constraints(m_p, p_amp=np.pi, v_amp=v_amp)
    A_u, _ = get_linear_box_constraints(m_u, amp=u_amp)
    n_x, n_u = A_x.shape[0], A_u.shape[0]
    weights = (1 / 0.1, 1 / v_amp, 1 / u_amp, (n_x + n_u) / m_p)

    time_s = time.time()
    pool_a, pool_v = sample_model_error_terms(problem, B, v_amp, nr_theta=pool_size[0], nr_q=pool_size[1], rng=rng)
    verif_a, verif_v = sample_model_error_terms(problem, B, v_amp, nr_theta=verif_size[0], nr_q=verif_size[1],
                                                include_vertices=True, rng=rng)
    verif_c = sample_disc_and_gravity_errors(problem, A, B, u_amp, v_amp, nr_samples=nr_disc, rng=rng)
    logger.debug(f"muestras: pool {pool_a.shape[0]}/{pool_v.shape[0]}, verificación {verif_a.shape[0]}/"
                 f"{verif_v.shape[0]}/{verif_c.shape[0]} ({time.time() - time_s:0.1f}s)")

    def evaluate(sol, rho, contraction):
        return evaluate_controller(sol["E"], sol["Y"], rho, verif_a, verif_v, verif_c, A_x, A_u, u_amp, v_amp,
                                   A=A, B=B, contraction=contraction)

    results = []
    for rho in rhos:
        base = solve_base(A, B, A_x, A_u, verts_w, rho, weights, solver)
        if base is None:
            logger.debug(f"base rho {rho:0.3f}: infactible")
            continue
        T = base["T"]
        P_b, K_b, P_sqrt_b, P_inv_sqrt_b = _controller_from_EY(base["E"], base["Y"])
        if "base" in methods:
            results.append(dict(method="base", rho=rho, rho_tilde_max=None, sol=base,
                                ev=evaluate(base, rho, "product"), hist=None))
        if "base_robust_analysis" in methods:
            results.append(dict(method="base_robust_analysis", rho=rho, rho_tilde_max=None, sol=base,
                                ev=evaluate(base, rho, "robust"), hist=None))

        if "lmi_robust" in methods:
            res = solve_robust(A, B, A_x, A_u, verts_w, pool_a, rho, weights, solver, T=T,
                               init_P_sqrt=P_sqrt_b, logger=logger)
            if res["sol"] is not None and res["status"] == "optimal":
                results.append(dict(method="lmi_robust", rho=rho, rho_tilde_max=None, sol=res["sol"],
                                    ev=evaluate(res["sol"], rho, "robust"), hist=None))
            else:
                logger.debug(f"lmi_robust rho {rho:0.3f}: {res['status']}")

        if "lmi_product" in methods:
            s_b, t_b = _s_t(K_b, P_inv_sqrt_b)
            ka_b, kv_b = norms_P(P_sqrt_b, pool_a).max(), norms_P(P_sqrt_b, pool_v).max()
            init_base = dict(P_sqrt=P_sqrt_b, lam_a=s_b / ka_b, lam_v=t_b / kv_b)
            ph1 = solve_lmi_ccp(A, B, A_x, A_u, verts_w, pool_a, pool_v, rho, weights, solver,
                                mode="rho_tilde", init=init_base, T=T, logger=logger)
            if ph1["sol"] is None:
                continue
            rt_min = ph1["hist"][-1]["rho_tilde_pool"]
            results.append(dict(method="lmi_product_min_rho_tilde", rho=rho, rho_tilde_max=None, sol=ph1["sol"],
                                ev=evaluate(ph1["sol"], rho, "product"), hist=ph1["hist"]))
            rt_base_pool = rho + ka_b * s_b + kv_b * t_b
            for rt_max in rho_tilde_grid:
                if rt_max <= rt_min + 1e-4:
                    continue
                if rt_max >= rt_base_pool:
                    init = dict(init_base)
                else:
                    sol1 = ph1["sol"]
                    init = dict(P_sqrt=sol1["P_sqrt"], lam_a=sol1["lam_a"], lam_v=sol1["lam_v"],
                                act_a=sol1["act_a"], act_v=sol1["act_v"])
                res = solve_lmi_ccp(A, B, A_x, A_u, verts_w, pool_a, pool_v, rho, weights, solver,
                                    mode="tightening", rho_tilde_max=rt_max, init=init, T=T, logger=logger)
                if res["status"] != "optimal" or res["sol"] is None:
                    logger.debug(f"lmi_product rho {rho:0.3f} rt_max {rt_max}: {res['status']}")
                    continue
                results.append(dict(method="lmi_product", rho=rho, rho_tilde_max=rt_max, sol=res["sol"],
                                    ev=evaluate(res["sol"], rho, "product"), hist=res["hist"]))
    logger.debug(f"pipeline LMI listo en {time.time() - time_s:0.1f}s")
    samples = dict(pool_a=pool_a, pool_v=pool_v, verif_a=verif_a, verif_v=verif_v, verif_c=verif_c, weights=weights)
    return results, samples


def pick_best(results, method):
    """Menor fitness entre los controladores del método dado con rho_tilde verificado < 1."""
    names = {method, method + "_min_rho_tilde"}   # la fase 1 también es un candidato de lmi_product
    cands = [r for r in results if r["method"] in names and r["ev"]["rho_tilde"] < 1]
    if not cands:
        return None
    return min(cands, key=lambda r: r["ev"]["fitness"])


def to_builtin(obj):
    """
    Convierte arreglos y escalares de numpy a tipos de Python. Así los .pckl se pueden abrir con
    cualquier versión de numpy (un pickle hecho con numpy 2 no se abre con numpy 1.26).
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_builtin(v) for v in obj)
    if isinstance(obj, set):
        return sorted(to_builtin(v) for v in obj)
    return obj


def save_results(path, results, samples):
    """Guarda la evaluación de toda la grilla. Las muestras (decenas de MB) no se guardan, solo cuántas fueron."""
    slim = []
    for r in results:
        ev = {k: v for k, v in r["ev"].items() if np.ndim(v) == 0}   # escalares: rho, rho_tilde, a, b, c, ...
        slim.append(dict(method=r["method"], rho=r["rho"], rho_tilde_max=r["rho_tilde_max"], ev=ev,
                         E=r["sol"]["E"], Y=r["sol"]["Y"],
                         hist=r["hist"] if isinstance(r["hist"], list) else None))
    n_samples = {k: (v.shape[0] if hasattr(v, "shape") else v) for k, v in samples.items()}
    with open(path, "wb") as fp:
        pickle.dump(to_builtin(dict(results=slim, n_samples=n_samples)), fp)


def summary_table(results, methods=("base", "base_robust_analysis", "lmi_product", "lmi_robust")):
    """Texto con el controlador elegido por cada método (mismo criterio de selección que los autores)."""
    lines = [
        f"{'método':22s} {'rho':>6s} {'rho_t':>7s} {'a':>8s} {'b':>8s} {'c':>8s} {'s':>7s} "
        f"{'delta_f':>8s} {'r_p*df[mrad]':>12s} {'fitness':>8s}"
    ]
    for m in methods:
        best = pick_best(results, m)
        if best is None:
            lines.append(f"{m:22s}  sin controlador con rho_tilde < 1")
            continue
        ev = best["ev"]
        extra = f"  (rho_tilde_max {best['rho_tilde_max']})" if best.get("rho_tilde_max") else ""
        lines.append(
            f"{m:22s} {best['rho']:6.3f} {ev['rho_tilde']:7.4f} {ev['a']:8.5f} {ev['b']:8.5f} {ev['c']:8.5f} "
            f"{ev['s']:7.3f} {ev['delta_f']:8.4f} {ev['delta_f_rad'] * 1e3:12.3f} {ev['fitness']:8.4f}{extra}"
        )
    return "\n".join(lines)
