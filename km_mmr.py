#!/usr/bin/env python3
"""
KM_por_MMR.py
Kaplan-Meier + log-rank + Cox por proteina MMR mutada.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import multivariate_logrank_test, pairwise_logrank_test
import warnings
warnings.filterwarnings("ignore")
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.plotting import add_at_risk_counts # <- NUEVO IMPORT
from lifelines.statistics import multivariate_logrank_test, pairwise_logrank_test

# ─── CONFIGURACION ────────────────────────────────────────────────────────────
SURV_FILE = Path.home() / "Desktop/tfm/datos/supervivencia.xlsx"
OUTDIR    = Path.home() / "Desktop/tfm/km_survival/resultados_KM_MMR_4"
OUTDIR.mkdir(parents=True, exist_ok=True)

MIN_N_PLOT = 3
ALPHA      = 0.05

PALETTE = {
    "MLH1": "#9C27B0",
    "MLH1/PMS2": "#29B6F6",
    "MSH2": "#1E88E5",
    "MSH2/MSH6": "#EF5350",
    "MSH6": "#E0A43A",
    "MSH6/PMS2/MLH1": "#8D6E63",
    "PMS2": "#66BB6A"
}

# ─── CARGA ────────────────────────────────────────────────────────────────────
df = pd.read_excel(SURV_FILE)
df.columns = df.columns.str.strip().str.lower()

mmr_col = next((c for c in df.columns if "mmr_proteina" in c or "mmr_grupo" in c), "mmr_grupo")
df = df.rename(columns={mmr_col: "mmr_grupo"})

df["mmr_grupo"] = df["mmr_grupo"].astype(str).str.strip()
df = df[~df["mmr_grupo"].isin(["nan", "desconocido", ""])]
df = df.dropna(subset=["time", "event", "mmr_grupo"])
df["time"]  = pd.to_numeric(df["time"], errors="coerce")
df["event"] = pd.to_numeric(df["event"], errors="coerce")
df = df.dropna(subset=["time", "event"])
df = df[df["time"] > 0].copy()

print("Pacientes con datos completos:", len(df))
print(df["mmr_grupo"].value_counts())

# ─── FILTRAR GRUPOS PEQUEÑOS ──────────────────────────────────────────────────
counts = df["mmr_grupo"].value_counts()
grupos_validos = counts[counts >= MIN_N_PLOT].index.tolist()
df = df[df["mmr_grupo"].isin(grupos_validos)].copy()

print("Grupos retenidos:", grupos_validos)

# ─── KAPLAN-MEIER ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 7))

# NUEVO: Crear una lista para ir guardando cada iteración del KaplanMeierFitter
fitters = []

for grupo in sorted(df["mmr_grupo"].unique()):
    sub = df[df["mmr_grupo"] == grupo]
    color = PALETTE.get(grupo, "#607D8B")
    kmf = KaplanMeierFitter()
    kmf.fit(sub["time"], event_observed=sub["event"], label=f"{grupo} (n={len(sub)})")
    kmf.plot_survival_function(ax=ax, color=color, linewidth=2.2, ci_show=True, ci_alpha=0.08)
    
    # NUEVO: Guardamos el modelo ajustado en la lista
    fitters.append(kmf)

result_global = multivariate_logrank_test(df["time"], df["mmr_grupo"], df["event"])
p_global = result_global.p_value

titulo = "Supervivencia global por MMR alterada\nLog-rank p = " + str(round(p_global, 4))
ax.set_title(titulo, fontsize=14, fontweight="bold", pad=14)
ax.set_xlabel("Tiempo (meses)", fontsize=12)
ax.set_ylabel("Probabilidad de supervivencia", fontsize=12)
ax.set_ylim(-0.02, 1.05)
ax.legend(loc="upper right", fontsize=10, framealpha=0.9)
ax.grid(axis="y", alpha=0.3, linestyle="--")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# NUEVO: Añadimos la tabla de pacientes en riesgo utilizando los fitters guardados
add_at_risk_counts(*fitters, ax=ax, fig=fig)

# Ajustar los márgenes para que la tabla inferior no se corte
plt.tight_layout()
out_km = OUTDIR / "KM_MMR_grupos.png"
fig.savefig(out_km, dpi=180, bbox_inches="tight")
plt.close()
print("Grafico guardado:", out_km)
# %%


# ─── PAIRWISE LOG-RANK ────────────────────────────────────────────────────────
pw = pairwise_logrank_test(df["time"], df["mmr_grupo"], df["event"])
pw_summary = pw.summary[["p"]].copy()
pw_summary.columns = ["p_value"]
pw_summary["significativo"] = pw_summary["p_value"] < ALPHA
pw_summary = pw_summary.sort_values("p_value")

out_pw = OUTDIR / "KM_MMR_pairwise.csv"
pw_summary.to_csv(out_pw)
print("Pairwise guardado:", out_pw)
print(pw_summary.to_string())

# ─── COX UNI Y MULTIVARIANTE ──────────────────────────────────────────────────
referencia = "MLH1/PMS2"
grupos_cox = [g for g in df["mmr_grupo"].unique() if g != referencia]

cox_rows = []
for grupo in grupos_cox:
    sub_cox = df[df["mmr_grupo"].isin([referencia, grupo])].copy()
    sub_cox["grupo_bin"] = (sub_cox["mmr_grupo"] == grupo).astype(int)
    if sub_cox["grupo_bin"].nunique() < 2:
        continue
    try:
        cph = CoxPHFitter()
        cph.fit(sub_cox[["time", "event", "grupo_bin"]], duration_col="time", event_col="event")
        s = cph.summary.loc["grupo_bin"]
        cox_rows.append({
            "grupo": grupo,
            "referencia": referencia,
            "n_grupo": int((sub_cox["mmr_grupo"] == grupo).sum()),
            "n_ref": int((sub_cox["mmr_grupo"] == referencia).sum()),
            "HR_uni": round(np.exp(s["coef"]), 3),
            "HR_lower_uni": round(np.exp(s["coef lower 95%"]), 3) if "coef lower 95%" in s.index else round(s.get("exp(coef) lower 95%", np.nan), 3),
            "HR_upper_uni": round(np.exp(s["coef upper 95%"]), 3) if "coef upper 95%" in s.index else round(s.get("exp(coef) upper 95%", np.nan), 3),
            "p_uni": round(s["p"], 4),
        })
    except Exception as e:
        print("Cox univariante fallo para", grupo, ":", e)

cox_multi = []
for grupo in grupos_cox:
    sub_cox = df[df["mmr_grupo"].isin([referencia, grupo])].copy()
    sub_cox["grupo_bin"] = (sub_cox["mmr_grupo"] == grupo).astype(int)
    sub_cox = sub_cox.dropna(subset=["age", "stage"])
    if len(sub_cox) < 10 or sub_cox["grupo_bin"].nunique() < 2:
        continue
    try:
        sub_cox["age"] = pd.to_numeric(sub_cox["age"], errors="coerce")
        sub_cox["stage"] = pd.to_numeric(sub_cox["stage"], errors="coerce")
        sub_cox = sub_cox.dropna(subset=["age", "stage"])
        cph = CoxPHFitter()
        cph.fit(sub_cox[["time", "event", "grupo_bin", "age", "stage"]], duration_col="time", event_col="event")
        s = cph.summary.loc["grupo_bin"]
        cox_multi.append({
            "grupo": grupo,
            "HR_adj": round(np.exp(s["coef"]), 3),
            "HR_lower_adj": round(np.exp(s["coef lower 95%"]), 3) if "coef lower 95%" in s.index else round(s.get("exp(coef) lower 95%", np.nan), 3),
            "HR_upper_adj": round(np.exp(s["coef upper 95%"]), 3) if "coef upper 95%" in s.index else round(s.get("exp(coef) upper 95%", np.nan), 3),
            "p_adj": round(s["p"], 4),
        })
    except Exception as e:
        print("Cox multivariante fallo para", grupo, ":", e)

df_cox = pd.DataFrame(cox_rows)
df_cox_multi = pd.DataFrame(cox_multi)
if not df_cox.empty and not df_cox_multi.empty:
    df_cox_final = df_cox.merge(df_cox_multi, on="grupo", how="left")
else:
    df_cox_final = df_cox if not df_cox.empty else df_cox_multi

out_cox = OUTDIR / "Cox_MMR.csv"
df_cox_final.to_csv(out_cox, index=False)
print("Cox guardado:", out_cox)
print(df_cox_final.to_string())

# ─── TABLA RESUMEN ────────────────────────────────────────────────────────────
resumen = []
for grupo in sorted(df["mmr_grupo"].unique()):
    sub = df[df["mmr_grupo"] == grupo]
    resumen.append({
        "mmr_grupo": grupo,
        "n": len(sub),
        "eventos": int(sub["event"].sum()),
        "mediana_tiempo_meses": round(sub["time"].median(), 1),
    })

df_resumen = pd.DataFrame(resumen).sort_values("n", ascending=False)
out_res = OUTDIR / "Resumen_grupos_MMR.csv"
df_resumen.to_csv(out_res, index=False)
print("Resumen grupos:")
print(df_resumen.to_string(index=False))
print("\nAnalisis completado.")