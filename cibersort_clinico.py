#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# =========================
# 1. ARCHIVOS
# =========================
ARCHIVO_CIBERSORT = "/Users/andreasantiago/Desktop/tfm/datos/CIBERSORT2/CIBERSORTx_Job_Results.csv"
ARCHIVO_GRUPOS    = "/Users/andreasantiago/Desktop/tfm/datos/MMR_proteina/grupos_MMR_proteina.csv"
CARPETA_SALIDA    = "/Users/andreasantiago/Desktop/tfm/datos/CIBERSORT2/"

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
if "MMR_proteina" not in grupos.columns:
    raise ValueError("No encuentro la columna MMR_proteina en el CSV de grupos")

# Normalizar IDs
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

grupos["Cluster"] = grupos["MMR_proteina"].astype(str).str.strip()

# =========================
# 3. CRUCE (una sola vez)
# =========================
df = pd.merge(cibersort, grupos[["ID", "Cluster"]], on="ID", how="inner")

print(f"Muestras cruzadas: {len(df)}")
print(f"Clusters encontrados: {df['Cluster'].unique().tolist()}")

# =========================
# 4. IDENTIFICAR COLUMNAS DE CÉLULAS
# =========================
exclude = {"ID", "P-value", "Pval", "Correlation", "RMSE", "Cluster"}
celulas = [c for c in cibersort.columns if c not in exclude]

if len(celulas) < 2:
    raise ValueError(f"Solo he detectado {len(celulas)} columnas celulares. Revisa el CSV.")

print(f"Células detectadas ({len(celulas)}): {celulas}")

# =========================
# 5. HEATMAP
# =========================
orden_grupos = ["MLH1/PMS2", "MLH1", "PMS2", "MSH2/MSH6", "MSH2", "MSH6", "MSH6/PMS2/MLH1"]


orden_grupos = [g for g in orden_grupos if g in df["Cluster"].unique()]
grupos_sin_orden = [g for g in df["Cluster"].unique() if g not in orden_grupos]
orden_grupos = orden_grupos + grupos_sin_orden 

print(f"Grupos en el heatmap: {orden_grupos}")

df["Cluster"] = pd.Categorical(df["Cluster"], categories=orden_grupos, ordered=True)

perfil_medio = df.groupby("Cluster", observed=True)[celulas].mean()

print(f"\nPerfil medio (shape): {perfil_medio.shape}")
print(perfil_medio)

# Normalización: 
perfil_norm = (perfil_medio - perfil_medio.mean()) / perfil_medio.std(ddof=0)
perfil_norm = perfil_norm.replace([np.inf, -np.inf], np.nan).fillna(0)

fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(
    perfil_norm.T,
    cmap="RdBu_r",
    center=0,
    linewidths=0.3,
    linecolor="white",
    ax=ax
)
ax.set_title("Perfil inmunológico medio por grupo MMR", fontsize=16, fontweight="bold")
ax.set_xlabel("Grupos")
ax.set_ylabel("Poblaciones celulares (CIBERSORTx)")
plt.tight_layout()
plt.savefig(CARPETA_SALIDA + "Heatmap_Cibersort_MMR.png", dpi=300, bbox_inches="tight")
plt.show(block=True)

# =========================
# 6. ESTADÍSTICA: KRUSKAL-WALLIS
# =========================
resultados_stats = []

for celula in celulas:
    grupos_vals = []
    for cluster in sorted(df["Cluster"].dropna().unique()):
        vals = df.loc[df["Cluster"] == cluster, celula].dropna().values
        if len(vals) > 0:
            grupos_vals.append(vals)

    if len(grupos_vals) >= 2:
        stat, p_val = stats.kruskal(*grupos_vals)
    else:
        stat, p_val = np.nan, np.nan

    resultados_stats.append({"Celula": celula, "P_valor": p_val})

df_stats = pd.DataFrame(resultados_stats).sort_values("P_valor")
df_stats["FDR"] = df_stats["P_valor"] * len(df_stats) / (df_stats["P_valor"].rank(method="first"))

df_stats.to_csv(CARPETA_SALIDA + "Resultados_Cibersort_MMR.csv", index=False)

sig = df_stats[df_stats["P_valor"] < 0.05]["Celula"].tolist()
print(f"\nCélulas significativas (p < 0.05): {len(sig)}")
print(sig)

# =========================
# 7. BOXPLOTS
# =========================
top_celulas = sig[:6]

if len(top_celulas) > 0:
    n_cols = min(3, len(top_celulas))
    n_rows = (len(top_celulas) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    axes = np.array(axes).reshape(-1)

    palette = sns.color_palette("Set2", n_colors=df["Cluster"].nunique())
    palette_dict = dict(zip(sorted(df["Cluster"].dropna().unique()), palette))

    for i, celula in enumerate(top_celulas):
        sns.boxplot(
            x="Cluster", y=celula, data=df,
            ax=axes[i], palette=palette_dict, showfliers=False
        )
        sns.stripplot(
            x="Cluster", y=celula, data=df,
            ax=axes[i], color="black", alpha=0.35, size=3
        )
        p_val = df_stats.loc[df_stats["Celula"] == celula, "P_valor"].values[0]
        axes[i].set_title(f"{celula}\np = {p_val:.4e}", fontsize=11, fontweight="bold")
        axes[i].set_xlabel("")
        axes[i].set_ylabel("Fracción celular")
        axes[i].tick_params(axis="x", rotation=15)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(CARPETA_SALIDA + "Boxplots_Cibersort_MMR.png", dpi=300, bbox_inches="tight")
    plt.show(block=True)
else:
    print("No hay células significativas con p < 0.05.")
    
# %%
    
# =========================
# 7. BOXPLOTS PARA TODOS LOS TIPOS CELULARES
# =========================

top_celulas = sig         # OPCIÓN A: Todas las células que hayan salido con p < 0.05
#top_celulas = celulas    # OPCIÓN B: Absolutamente TODAS las células de CIBERSORTx (sean significativas o no)

if len(top_celulas) > 0:
    n_cols = min(4, len(top_celulas)) 
    n_rows = (len(top_celulas) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]

    palette = sns.color_palette("Set2", n_colors=df["Cluster"].nunique())
    palette_dict = dict(zip(sorted(df["Cluster"].dropna().unique()), palette))

    for i, celula in enumerate(top_celulas):
        sns.boxplot(
            x="Cluster", y=celula, data=df,
            ax=axes[i], palette=palette_dict, showfliers=False
        )
        sns.stripplot(
            x="Cluster", y=celula, data=df,
            ax=axes[i], color="black", alpha=0.35, size=3
        )
        
        p_val_array = df_stats.loc[df_stats["Celula"] == celula, "P_valor"].values
        p_val = p_val_array[0] if len(p_val_array) > 0 else np.nan
        
        axes[i].set_title(f"{celula}\np = {p_val:.4f}", fontsize=11, fontweight="bold")
        axes[i].set_xlabel("")
        axes[i].set_ylabel("Fracción celular")
        axes[i].tick_params(axis="x", rotation=15)


    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(CARPETA_SALIDA + "Boxplots_Cibersort_MMR_Todas_2.png", dpi=300, bbox_inches="tight")
    plt.show(block=True)
else:
    print("No hay células para graficar en esta selección.")
# %%

# =========================
# 5b. HEATMAP POR MUESTRAS INDIVIDUALES
# =========================

df_muestras = df.sort_values("Cluster").reset_index(drop=True)

matriz_muestras = df_muestras[celulas].astype(float)

# Normalización Z-score por célula: 
matriz_muestras_norm = (matriz_muestras - matriz_muestras.mean()) / matriz_muestras.std(ddof=0)
matriz_muestras_norm = matriz_muestras_norm.replace([np.inf, -np.inf], np.nan).fillna(0)

df_muestras["Cluster_str"] = df_muestras["Cluster"].astype(str)

unique_clusters = [c for c in orden_grupos if c in df_muestras["Cluster_str"].values]
palette_muestras = sns.color_palette("Set2", n_colors=len(unique_clusters))
cluster_color_dict = dict(zip(unique_clusters, palette_muestras))

colores_muestras = df_muestras["Cluster_str"].map(cluster_color_dict)

g = sns.clustermap(
    matriz_muestras_norm.T,
    row_cluster=True,      
    col_cluster=False,     
    col_colors=colores_muestras,
    cmap="RdBu_r",
    center=0,
    figsize=(16, 12),
    cbar_kws={'label': 'Z-score (Abundancia relativa)'},
    xticklabels=False     
)

g.ax_heatmap.set_title("Perfiles inmunológicos por muestra individual", fontsize=16, fontweight="bold", pad=120)
g.ax_heatmap.set_xlabel(f"Muestras individuales (n = {len(df_muestras)})", fontsize=12)
g.ax_heatmap.set_ylabel("Poblaciones celulares", fontsize=12)


from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=cluster_color_dict[clus], label=clus) for clus in unique_clusters]
g.ax_col_colors.legend(
    handles=legend_elements, 
    title="Grupos MMR", 
    loc='lower center', 
    bbox_to_anchor=(0.5, 1.2), 
    ncol=min(len(unique_clusters), 4) )

plt.savefig(CARPETA_SALIDA + "Heatmap_Muestras_Individuales_MMR.png", dpi=300, bbox_inches="tight")
plt.show(block=True)
