# Extensión IEE3002 (Grupo 24): diseño LMI del controlador auxiliar

Todo lo nuevo está en archivos aparte. El código de los autores solo tiene parches mínimos, marcados con `IEE3002`.

## Archivos nuevos

| Archivo | Qué hace |
|---|---|
| `offline_pipeline_b2_lmi_controller.py` (raíz) | Los SDP: método base acoplado (4×4), formulación (24) con CCP y la variante con contracción robusta. También el muestreo del error de modelo, la evaluación común y la selección por *fitness*. |
| `1b_run_offline_lmi.py` | Etapa offline para un nivel de incertidumbre. Guarda la grilla completa en `data/dof_2_ef_{ef}_lmi_results.pckl` y una carpeta por método, `data/dof_2_ef_{ef}_{método}/`, lista para `2_run_all.py`. |
| `1c_verify_dense_grid.py` | Verifica κ_a, κ_v y ρ̃ de un controlador guardado con una grilla densa sobre (q2, m1, m2). |
| `3b_collect_sweep.py` | Junta el barrido de incertidumbre en `data/sweep_summary.csv` y en `figs/`. |

## Métodos que se comparan

Todos se evalúan con las mismas muestras de verificación, independientes de las usadas para diseñar, y se eligen con el mismo criterio que los autores: menor *fitness* entre los controladores con ρ̃ < 1.

- **`base`**: SDP (11) y ρ̃ = ρ + κ_a s + κ_v t, como en el paper (ec. (15)).
- **`base_robust_analysis`**: el mismo (P, K) que `base`, pero con ρ̃ calculado con la cota de contracción robusta. Sirve para separar cuánto de la mejora viene del análisis y cuánto del diseño.
- **`lmi_product`**: formulación (24) del informe de avance (κ², s² y t² como variables, y cota con λ y CCP).
- **`lmi_robust`**: SDP (11) con la contracción robusta impuesta en las muestras de B M̃: `||A + BK − B M̃_l K||_P ≤ ρ`. Con eso, ρ̃ = ρ + κ_v t.

## Cómo correrlo (desde la raíz del repo)

```bash
python examples/planar_two_dof/1b_run_offline_lmi.py --error-frac 0.2           # ~3 min
PYTHONPATH=. python examples/planar_two_dof/2_run_all.py dof_2_ef_0.2_lmi_robust --only ft
python examples/planar_two_dof/1c_verify_dense_grid.py dof_2_ef_0.2_lmi_robust
python examples/planar_two_dof/3b_collect_sweep.py
```

Para un `error_frac` sin la etapa "a" de los autores, `1b` la calcula con muestreo rápido. Usa MOSEK si está instalado; si no, Clarabel.

## Parches a archivos de los autores

- `problem_scenario.py` → `from_cached_dir` usa la carpeta actual y, opcionalmente, el URDF local, aunque `ps.pckl` venga de otro computador.
- `examples/planar_two_dof/2_run_all.py` → recibe por argumento la carpeta y un `--only` con los métodos a correr. También salta los controladores que faltan.
- `controllers/flexible_tube_controller.py` y `controllers/tube_controller.py` → `eigvalsh` en vez de `eig` para r_p. Con una P acoplada, `eig` devolvía complejos y el SOCP fallaba.
- `path_planner/rrt_bi_dir.py` → `simple_shortcut` calcula fila por fila (memoria O(N)); da el mismo resultado.

## Resultados del 15/11 (Python 3.13, Clarabel; los tiempos de cómputo hay que medirlos en la Raspberry)

Con este ambiente, las 10 corridas `ft` del método de los autores reproducen exactamente su `results.csv`.

Offline con ±20 %, mismo estimador para todos:

| Método | ρ | ρ̃ | δ_f | r_p·δ_f [mrad] | fitness |
|---|---|---|---|---|---|
| base | 0,90 | 0,9469 | 0,0466 | 0,494 | 0,088 |
| base + análisis robusto | 0,90 | 0,9309 | 0,0358 | 0,379 | 0,068 |
| formulación (24) | 0,90 | 0,9467 | 0,0466 | 0,494 | 0,087 |
| LMI robusto | 0,90 | 0,9012 | 0,0260 | 0,273 | 0,052 |

Barrido de incertidumbre: tabla completa en `data/sweep_summary.csv` y figuras en `figs/`.

| ± masas | base | base + análisis robusto | LMI robusto |
|---|---|---|---|
| 10 % | 2,81 s | 2,84 s | 2,81 s |
| 20 % | 2,96 s | 2,91 s | 2,91 s |
| 30 % | 3,72 s | 3,32 s | 3,12 s |
| 35 % | 4,50 s | 3,74 s | 3,30 s |
| 40 % | sin controlador (ρ̃ ≥ 1) | 4,14 s | 3,59 s |
| 60 % | — | 5,51 s | 4,28 s |
| 80 % | — | 8,82 s (2/10 llegan) | 5,79 s |
| 90 % | — | sin controlador | 6,86 s |

Todas las corridas listadas llegaron a la meta (10/10), salvo donde se indica.
