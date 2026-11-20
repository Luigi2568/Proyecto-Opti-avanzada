#!/bin/bash
# IEE3002 (Grupo 24): corre todo el barrido en la Raspberry. Desde la raíz del repo, con el venv activado:
#   nohup bash examples/planar_two_dof/correr_todo_rpi.sh > correr_todo.log 2>&1 &
# Sigue el avance con:  tail -f correr_todo.log
# sin "set -e": si un nivel falla, el barrido sigue con el resto
export PYTHONPATH=.
D=examples/planar_two_dof
NIVELES="0.1 0.2 0.3 0.35 0.4 0.5 0.6 0.7 0.8 0.9"

echo "=== 1. Etapa offline por nivel ($(date))"
for ef in $NIVELES; do
  if [ "$ef" = "0.2" ]; then
    METODOS="base base_robust_analysis lmi_product lmi_robust"   # (24) solo donde importa
  else
    METODOS="base base_robust_analysis lmi_robust"
  fi
  echo "--- offline ef=$ef ($(date +%H:%M))"
  python -W ignore $D/1b_run_offline_lmi.py --error-frac $ef --methods $METODOS
done

echo "=== 2. Simulación en línea (solo ft) ($(date))"
for ef in $NIVELES; do
  for m in base base_robust_analysis lmi_robust; do
    d=dof_2_ef_${ef}_$m
    if [ -f $D/data/$d/ftube_controller.npz ]; then
      echo "--- en línea $d ($(date +%H:%M))"
      python -W ignore $D/2_run_all.py $d --only ft
    fi
  done
done

echo "=== 3. Resumen ($(date))"
python $D/3b_collect_sweep.py
python $D/3c_solve_times.py dof_2_ef_0.2 dof_2_ef_0.2_base dof_2_ef_0.2_lmi_robust \
       dof_2_ef_0.35_base dof_2_ef_0.35_lmi_robust dof_2_ef_0.9_lmi_robust
echo "=== listo ($(date))"
