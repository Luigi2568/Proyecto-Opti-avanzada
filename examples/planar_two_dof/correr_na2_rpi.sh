#!/bin/bash
# IEE3002 (Grupo 24): experimento con ejecución en paralelo n_a = 2 (presupuesto de 20 ms por MPC).
# Desde la raíz del repo, con el venv activado:
#   nohup bash examples/planar_two_dof/correr_na2_rpi.sh > correr_na2.log 2>&1 &
# No pisa los resultados con n_a = 1: escribe results_na2.csv y motion_na2/.
export PYTHONPATH=.
D=examples/planar_two_dof
CARPETAS="dof_2_ef_0.2 dof_2_ef_0.2_base dof_2_ef_0.2_lmi_robust dof_2_ef_0.35_base dof_2_ef_0.35_lmi_robust dof_2_ef_0.9_lmi_robust"

for d in $CARPETAS; do
  echo "--- n_a = 2: $d ($(date +%H:%M))"
  python -W ignore $D/2_run_all.py $d --only ft --n-a 2
done

echo "=== tiempo a la meta: n_a = 1 vs n_a = 2"
python $D/3d_compare_na.py $CARPETAS --n-a 2
echo "=== tiempos de cómputo con n_a = 2 (límite 20 ms)"
python $D/3c_solve_times.py $CARPETAS --n-a 2
echo "=== listo ($(date))"
