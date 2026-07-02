#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 09:47:05 2026

@author: andreasantiago
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests 
import warnings
warnings.filterwarnings('ignore')

# =========================
# 1. ARCHIVOS
# =========================
ARCHIVO_CIBERSORT = "/Users/andreasantiago/Desktop/tfm/datos/CIBERSORT2/CIBERSORTx_Job_Results.csv"
ARCHIVO_GRUPOS    = "/Users/andreasantiago/Desktop/tfm/datos/LYNCH/grupos_Lynch.csv"
CARPETA_SALIDA    = "/Users/andreasantiago/Desktop/tfm/datos/LYNCH/CIBERSORT2/" 

# =========================
# 2. CARGA
# =========================
cibersort = pd.read_csv(ARCHIVO_CIBERSORT)

if "Mixture" in cibersort.columns:
    cibersort = cibersort.rename(columns={"Mixture": "ID"})

if "ID" not in cibersort.columns:
    raise ValueError("No encuentro la columna ID o Mixture en el CSV de CIBERSORTx")

grupos = pd.read_csv(ARCHIVO_GRUPOS)

if "CODE" not in grupos.columns:
    raise ValueError("No encuentro la columna CODE en el CSV de grupos")

if "LYNCH_SYNDROME_DICHOTOMIC" not in grupos.columns:
    raise ValueError("No encuentro la columna LYNCH_SYNDROME_DICHOTOMIC en el CSV de grupos")

# Normalizar IDs:
cibersort["ID"] = (
    cibersort["ID"].astype(str)
    .str.strip()
    .str.replace(" ", "_", regex=False)
)

grupos["ID"] = (
    grupos["CODE"].astype(str)
    .str.strip()
    .str.replace(" ", "_", regex=False)
)

grupos["Cluster"] = grupos["LYNCH_SYNDROME_DICHOTOMIC"].astype(str).str.strip()


df = pd.merge(cibersort, grupos[["ID", "Cluster"]], on="ID", how="inner")

# =========================
# 3. IDENTIFICAR COLUMNAS DE CÉLULAS
# =========================
exclude = {"ID", "P-value", "Pval", "Correlation", "RMSE", "Cluster"}
celulas = [c for c in cibersort.columns if c not in exclude]

if len(celulas) < 10:
    raise ValueError(f"Solo he detectado {len(celulas)} columnas celulares. Revisa el CSV de CIBERSORTx.")

# =========================
# 4. CRUCE Y RESUMEN
# =========================
print(f"Muestras cruzadas: {len(df)}")
print(f"Clusters: {df['Cluster'].unique().tolist()}")
print(f"Células detectadas: {len(celulas)}")
# %%


# =========================
# 5. HEATMAP
# =========================
perfil_medio = df.groupby("Cluster")[celulas].mean()

perfil_norm = (perfil_medio - perfil_medio.mean()) / perfil_medio.std(ddof=0)
perfil_norm = perfil_norm.replace([np.inf, -np.inf], np.nan).fillna(0)

plt.figure(figsize=(12, 10))
sns.heatmap(
    perfil_norm.T,
    cmap="RdBu_r",
    center=0,
    linewidths=0.3,
    linecolor="white"
)
plt.title("Perfil inmunológico medio por presencia de Síndrome de Lynch", fontsize=16, fontweight="bold")
plt.xlabel("Grupos")
plt.ylabel("Poblaciones celulares (CIBERSORTx)")
plt.tight_layout()
plt.savefig(CARPETA_SALIDA + "Heatmap_Cibersort_LYNCH_SYNDROME_DICHOTOMIC.png", dpi=300, bbox_inches="tight")
plt.show()



# =========================
# 6. ESTADÍSTICA: MANN-WHITNEY (Para 2 Grupos)
# =========================
resultados_stats = []
grupos_nombres = sorted(df["Cluster"].dropna().unique())

if len(grupos_nombres) != 2:
    print("Advertencia: Detectados", len(grupos_nombres), "grupos. El test de Mann-Whitney es para 2 grupos exactos.")

for celula in celulas:
    if len(grupos_nombres) >= 2:
        vals_g1 = df.loc[df["Cluster"] == grupos_nombres[0], celula].dropna().values
        vals_g2 = df.loc[df["Cluster"] == grupos_nombres[1], celula].dropna().values
        
        if len(vals_g1) > 0 and len(vals_g2) > 0:
            stat, p_val = stats.mannwhitneyu(vals_g1, vals_g2, alternative='two-sided')
        else:
            stat, p_val = np.nan, np.nan
    else:
        stat, p_val = np.nan, np.nan

    resultados_stats.append({"Celula": celula, "P_valor": p_val})

df_stats = pd.DataFrame(resultados_stats).sort_values("P_valor")

# Ajuste oficial FDR (Benjamini-Hochberg)
df_stats = df_stats.dropna(subset=['P_valor'])
_, fdr, _, _ = multipletests(df_stats["P_valor"], alpha=0.05, method='fdr_bh')
df_stats["FDR"] = fdr

df_stats.to_csv(CARPETA_SALIDA + "Resultados_Cibersort_LYNCH.csv", index=False)

sig = df_stats[df_stats["P_valor"] < 0.05]["Celula"].tolist()
print(f"Células significativas (p < 0.05): {len(sig)}")
# %%


# =========================
# 7. BOXPLOTS
# =========================
top_celulas = sig[:6]

if len(top_celulas) > 0:
    n_cols = min(3, len(top_celulas))
    n_rows = (len(top_celulas) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    
    # Prevenir error si solo hay 1 gráfica
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.reshape(-1)

    palette = sns.color_palette("Set2", n_colors=df["Cluster"].nunique())
    palette_dict = dict(zip(sorted(df["Cluster"].dropna().unique()), palette))

    for i, celula in enumerate(top_celulas):
        sns.boxplot(
            x="Cluster",
            y=celula,
            data=df,
            ax=axes[i],
            palette=palette_dict,
            showfliers=False
        )
        sns.stripplot(
            x="Cluster",
            y=celula,
            data=df,
            ax=axes[i],
            color="black",
            alpha=0.35,
            size=3
        )
        p_val = df_stats.loc[df_stats["Celula"] == celula, "P_valor"].values[0]
        axes[i].set_title(f"{celula}\np = {p_val:.4e}", fontsize=11, fontweight="bold")
        axes[i].set_xlabel("")
        axes[i].set_ylabel("Fracción celular")
        axes[i].tick_params(axis="x", rotation=15)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(CARPETA_SALIDA + "Boxplots_Cibersort_LYNCH.png", dpi=300, bbox_inches="tight")
    plt.show()
else:
    print("No hay células significativas con p < 0.05.")