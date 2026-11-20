# Extensión IEE3002 (Grupo 24): junta el barrido de incertidumbre (offline + online) en una tabla y figuras.
#
# Requiere haber corrido, para cada nivel:
#   python examples/planar_two_dof/1b_run_offline_lmi.py --error-frac EF
#   python examples/planar_two_dof/2_run_all.py dof_2_ef_EF_{método} --only ft
# Uso:  python examples/planar_two_dof/3b_collect_sweep.py
import pathlib
import pickle
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from examples.planar_two_dof import P_EXAMPLE_2_DOF
from offline_pipeline_b2_lmi_controller import pick_best

P_DATA = P_EXAMPLE_2_DOF / "data"
P_FIGS = P_EXAMPLE_2_DOF / "figs"
METHODS = ["base", "base_robust_analysis", "lmi_robust"]
LABELS = {"base": "Base (paper)", "base_robust_analysis": "Base + análisis robusto",
          "lmi_robust": "Diseño LMI robusto"}
COLORS = {"base": "#2a78d6", "base_robust_analysis": "#eb6834", "lmi_robust": "#1baf7a"}
MARKERS = {"base": "o", "base_robust_analysis": "s", "lmi_robust": "^"}


def collect():
    rows = []
    for f in sorted(P_DATA.glob("dof_2_ef_*_lmi_results.pckl")):
        ef = float(re.search(r"dof_2_ef_([0-9.]+)_lmi_results", f.name).group(1))
        with f.open("rb") as fp:
            results = pickle.load(fp)["results"]
        for m in METHODS + ["lmi_product"]:
            row = dict(error_frac=ef, method=m)
            best = pick_best(results, m)
            if best is not None:
                ev = best["ev"]
                row.update(rho=best["rho"], rho_tilde=ev["rho_tilde"], a=ev["a"], b=ev["b"], c=ev["c"],
                           delta_f=ev["delta_f"], delta_f_mrad=ev["delta_f_rad"] * 1e3, fitness=ev["fitness"])
            p_csv = P_DATA / f"dof_2_ef_{ef}_{m}" / "results.csv"
            if p_csv.exists():
                df = pd.read_csv(p_csv)
                df = df[df.name == "ft"]
                ok = df[df.status == "success"]
                row.update(n_runs=len(df), n_success=len(ok), t2g_mean_s=ok.t2g.mean() / 100,
                           t2g_std_s=ok.t2g.std() / 100,
                           statuses=";".join(f"{k}:{v}" for k, v in df.status.value_counts().items()))
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["error_frac", "method"])


def oracle_mean():
    p = P_DATA / "dof_2_ef_0.2" / "results.csv"
    df = pd.read_csv(p)
    ok = df[(df.name == "nom_star") & (df.status == "success")]
    return ok.t2g.mean() / 100


def plot(df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    P_FIGS.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

    # (a) contracción del tubo flexible vs incertidumbre
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for m in METHODS:
        d = df[(df.method == m) & df.rho_tilde.notna()]
        ax.plot(d.error_frac * 100, d.rho_tilde, marker=MARKERS[m], color=COLORS[m], lw=2, ms=6, label=LABELS[m])
    ax.axhline(1.0, color="#6b6b66", lw=1, ls="--")
    ax.text(40, 0.997, r"$\tilde\rho = 1$: no hay tubo flexible", fontsize=8, color="#6b6b66", ha="left", va="top")
    ax.set_xlabel("Incertidumbre en las masas [±%]")
    ax.set_ylabel(r"Contracción del tubo $\tilde\rho$")
    ax.grid(axis="y", color="#e5e5e2", lw=0.8)
    ax.legend(frameon=False, fontsize=8, loc="center right", bbox_to_anchor=(1.0, 0.42))
    fig.tight_layout()
    fig.savefig(P_FIGS / "sweep_rho_tilde.png", dpi=200)
    plt.close(fig)

    # (b) tiempo a la meta (corredores exitosos) vs incertidumbre
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for m in METHODS:
        d = df[(df.method == m) & df.t2g_mean_s.notna()]
        ax.plot(d.error_frac * 100, d.t2g_mean_s, marker=MARKERS[m], color=COLORS[m], lw=2, ms=6, label=LABELS[m])
        for _, r in d.iterrows():
            if r.n_success < r.n_runs:
                ax.annotate(f"{int(r.n_success)}/{int(r.n_runs)}", (r.error_frac * 100, r.t2g_mean_s),
                            textcoords="offset points", xytext=(4, 4), fontsize=7, color="#3d3d3a")
    t_or = None
    try:
        t_or = oracle_mean()
        ax.axhline(t_or, color="#6b6b66", lw=1, ls="--")
        ax.text(df.error_frac.max() * 100, t_or - 0.08, "Oráculo (modelo perfecto)", fontsize=8, color="#6b6b66",
                ha="right", va="top")
    except FileNotFoundError:
        pass
    ax.set_xlabel("Incertidumbre en las masas [±%]")
    ax.set_ylabel("Tiempo a la meta medio [s]")
    if t_or is not None:
        ax.set_ylim(bottom=t_or - 0.45)
    ax.grid(axis="y", color="#e5e5e2", lw=0.8)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(P_FIGS / "sweep_t2g.png", dpi=200)
    plt.close(fig)


def main():
    df = collect()
    df.to_csv(P_DATA / "sweep_summary.csv", index=False)
    cols = ["error_frac", "method", "rho", "rho_tilde", "delta_f_mrad", "fitness", "n_success", "t2g_mean_s"]
    with pd.option_context("display.width", 160, "display.max_rows", 200):
        print(df[[c for c in cols if c in df.columns]].to_string(index=False, float_format=lambda x: f"{x:0.4f}"))
    plot(df)
    print(f"tabla: {P_DATA / 'sweep_summary.csv'} | figuras: {P_FIGS}")


if __name__ == "__main__":
    main()
