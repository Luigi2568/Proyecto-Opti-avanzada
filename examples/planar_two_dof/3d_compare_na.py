# Extensión IEE3002 (Grupo 24): compara el tiempo a la meta con n_a = 1 y n_a = 2 (ejecución en paralelo).
#
# Uso:  PYTHONPATH=. python examples/planar_two_dof/3d_compare_na.py dof_2_ef_0.2 dof_2_ef_0.2_lmi_robust
import argparse
import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from examples.planar_two_dof import P_EXAMPLE_2_DOF


def summary(p_csv, method):
    if not p_csv.exists():
        return None
    df = pd.read_csv(p_csv)
    df = df[df.name == method]
    ok = df[df.status == "success"]
    return len(ok), len(df), ok.t2g.mean() / 100


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dir_names", nargs="+")
    parser.add_argument("--method", default="ft")
    parser.add_argument("--n-a", type=int, default=2)
    args = parser.parse_args()
    print(f"{'carpeta':30s} | {'n_a = 1':>16s} | {f'n_a = {args.n_a}':>16s} | diferencia")
    for name in args.dir_names:
        p_d = P_EXAMPLE_2_DOF / "data" / name
        s1 = summary(p_d / "results.csv", args.method)
        s2 = summary(p_d / f"results_na{args.n_a}.csv", args.method)
        fmt = lambda s: "—" if s is None else f"{s[0]}/{s[1]}  {s[2]:.3f} s"
        diff = "" if (s1 is None or s2 is None) else f"{(s2[2] - s1[2]):+.3f} s ({(s2[2] / s1[2] - 1) * 100:+.1f} %)"
        print(f"{name:30s} | {fmt(s1):>16s} | {fmt(s2):>16s} | {diff}")


if __name__ == "__main__":
    main()
