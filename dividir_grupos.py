#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 10:41:03 2026

@author: andreasantiago
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. CARGA Y EXPLORACIÓN DE DATOS
# =============================================================================

ARCHIVO_CLINICO = "/Users/andreasantiago/Desktop/tfm/datos/Datos_MSI_filtrado.xlsx"
CARPETA_SALIDA  = "/Users/andreasantiago/Desktop/tfm/datos/MMR_proteina/"

import os
os.makedirs(CARPETA_SALIDA, exist_ok=True)

df = pd.read_excel(ARCHIVO_CLINICO)

COLUMNA_MMR = 'MMR_proteina'  

print(f"\nValores únicos en '{COLUMNA_MMR}':")
print(df[COLUMNA_MMR].value_counts(dropna=False))

df = df[df[COLUMNA_MMR].notna() & (df[COLUMNA_MMR] != '')]
df[COLUMNA_MMR] = df[COLUMNA_MMR].astype(str).str.strip()

print(f"\nMuestras con proteína asignada: {len(df)}")
print(f"Grupos: {sorted(df[COLUMNA_MMR].unique())}")
# %%


# =============================================================================
# 2. GUARDAR CSV CON LAS MUESTRAS POR GRUPO
# =============================================================================
grupos_mmr = df[['CODE', COLUMNA_MMR]].copy()
grupos_mmr.to_csv(CARPETA_SALIDA + 'grupos_MMR_proteina.csv', index=False)

print("\n" + "="*55)
print("  MUESTRAS POR PROTEÍNA MMR MUTADA")
print("="*55)
for proteina in sorted(df[COLUMNA_MMR].unique()):
    muestras = df[df[COLUMNA_MMR] == proteina]['CODE'].tolist()
    print(f"\n{proteina}  ({len(muestras)} pacientes)")
    print("-" * 40)
    for m in muestras:
        print(f"  · {m}")
print("\n" + "="*55)

# %%


# =============================================================================
# 3. GRÁFICO 1 — DISTRIBUCIÓN DE MUESTRAS POR PROTEÍNA (barras + pie)
# =============================================================================
conteos = df[COLUMNA_MMR].value_counts()

colores_proteina = {
    'MLH1/PMS2':      '#A9BEDB',  # azul humo pastel
    'MSH2/MSH6':      '#A4A7CC',  # violeta azulado suave
    'PMS2':           '#9FC5B8',  # verde salvia agua
    'MSH6':           '#8FD0CF',  # turquesa menta suave
    'MLH1':           '#C7B6E6',  # lila empolvado
    'MSH2':           '#B9C3CC',  # gris azulado claro
    'MSH6/PMS2/MLH1': '#A8CFC8'   # verde agua grisáceo
}
colores = [colores_proteina.get(p, '#888888') for p in conteos.index]


plt.figure(figsize=(10, 6))

plt.bar(
    conteos.index,
    conteos.values,
    color=colores,
    edgecolor='white',
    linewidth=1.0
)

for i, (prot, n) in enumerate(zip(conteos.index, conteos.values)):
    plt.text(i, n + 0.8, str(n), ha='center', va='bottom',
             fontsize=12, fontweight='bold', color='#222222')

plt.title('Distribución de muestras por proteína MMR mutada',
          fontsize=16, fontweight='bold', pad=15)
plt.xlabel('Proteína MMR mutada', fontsize=12, fontweight='bold')
plt.ylabel('Número de muestras', fontsize=12, fontweight='bold')

plt.xticks(rotation=25, ha='right', fontsize=11)
plt.yticks(fontsize=11)
plt.ylim(0, conteos.max() * 1.15)

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#444444')
ax.spines['bottom'].set_color('#444444')
ax.grid(False)

plt.tight_layout()
plt.savefig(CARPETA_SALIDA + 'Distribucion_MMR_proteina_barras.png',
            dpi=300, bbox_inches='tight')
plt.show()



# %%

# =============================================================================
# 3. GRÁFICO 1 — DISTRIBUCIÓN DE MUESTRAS POR PROTEÍNA (solo barras)
# =============================================================================
conteos = df[COLUMNA_MMR].value_counts()

colores_relleno = {
    'MLH1/PMS2':      '#B4CBE8',
    'MSH2/MSH6':      '#B2B6DC',
    'PMS2':           '#AED2C6',
    'MSH6':           '#9EDEDC',
    'MLH1':           '#D2C0EE',
    'MSH2':           '#C3CDD6',
    'MSH6/PMS2/MLH1': '#B1D8D1'
}

colores_borde = {
    'MLH1/PMS2':      '#4A78B5',
    'MSH2/MSH6':      '#5E63B3',
    'PMS2':           '#4F9B86',
    'MSH6':           '#1AA7A1',
    'MLH1':           '#8E73C9',
    'MSH2':           '#7D92A5',
    'MSH6/PMS2/MLH1': '#5FA89F'
}

rellenos = [colores_relleno.get(p, '#CCCCCC') for p in conteos.index]
bordes = [colores_borde.get(p, '#666666') for p in conteos.index]

plt.figure(figsize=(10, 6))

plt.bar(
    conteos.index,
    conteos.values,
    color=rellenos,
    edgecolor=bordes,
    linewidth=2.0
)

for i, (prot, n) in enumerate(zip(conteos.index, conteos.values)):
    plt.text(
        i, n + 0.8, str(n),
        ha='center', va='bottom',
        fontsize=12, fontweight='bold', color='#222222'
    )

plt.title('Distribución de muestras por proteína MMR mutada',
          fontsize=16, fontweight='bold', pad=15)
plt.xlabel('Proteína MMR mutada', fontsize=12, fontweight='bold')
plt.ylabel('Número de muestras', fontsize=12, fontweight='bold')

plt.xticks(rotation=25, ha='right', fontsize=11)
plt.yticks(fontsize=11)
plt.ylim(0, conteos.max() * 1.15)

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#444444')
ax.spines['bottom'].set_color('#444444')
ax.grid(False)

plt.tight_layout()
plt.savefig(CARPETA_SALIDA + 'Distribucion_MMR_proteina_barras.png',
            dpi=300, bbox_inches='tight')
plt.show()


# %%

## NO INCLUIDA EN TFM 

# =============================================================================
# 4. GRÁFICO 2 — VARIABLES CLÍNICAS CLAVE POR PROTEÍNA (boxplots)
# =============================================================================
variables_continuas = ['DIAGNOSIS_AGE', 'STAGING_AJCC_DIAGNOSIS']
nombres_bonitos = {
    'DIAGNOSIS_AGE':           'Edad al diagnóstico',
    'STAGING_AJCC_DIAGNOSIS':  'Estadio AJCC',
}

fig, axes = plt.subplots(1, len(variables_continuas), figsize=(7 * len(variables_continuas), 6))
if len(variables_continuas) == 1:
    axes = [axes]

for ax, var in zip(axes, variables_continuas):
    if var not in df.columns:
        ax.set_visible(False)
        continue
    orden = sorted(df[COLUMNA_MMR].unique())
    col_lista = [colores_proteina.get(p, '#add8e6') for p in orden]
    sns.boxplot(x=COLUMNA_MMR, y=var, data=df, ax=ax,
                order=orden, palette=col_lista, showfliers=False)
    sns.stripplot(x=COLUMNA_MMR, y=var, data=df, ax=ax,
                  order=orden, color='black', alpha=0.35, size=3)

    # Kruskal-Wallis
    grupos_vals = [df[df[COLUMNA_MMR] == g][var].dropna().values for g in orden]
    grupos_vals = [g for g in grupos_vals if len(g) >= 2]
    if len(grupos_vals) >= 2:
        stat, p = stats.kruskal(*grupos_vals)
        ax.set_title(f"{nombres_bonitos.get(var, var)}\n(Kruskal-Wallis p = {p:.4f})", fontsize=12, fontweight='bold')
    else:
        ax.set_title(nombres_bonitos.get(var, var), fontsize=12, fontweight='bold')

    ax.set_xlabel('')
    ax.set_ylabel(nombres_bonitos.get(var, var))
    ax.tick_params(axis='x', rotation=25)

plt.tight_layout()
plt.savefig(CARPETA_SALIDA + 'Boxplots_clinicos_MMR.png', dpi=300, bbox_inches='tight')
plt.show()
print("  → Boxplots clínicos guardados.")
# %%

## NO INCLUIDA EN TFM

# =============================================================================
# 5. GRÁFICO 3 — VARIABLES CATEGÓRICAS (barras apiladas %)
# =============================================================================
variables_cat = {
    'KRAS_STATUS':  {0: 'No KRAS mutado', 1: 'KRAS mutado'},
    'LYNCH_SYNDROME_DICHOTOMIC': {0: 'No Lynch', 1: 'Síndrome de Lynch', 2: 'Síndrome de Lynch'},
    'MLH1_PROMOTOR_METHYLATION_RESULTS': {0: 'No metilado', 1: 'Metilado MLH1'},
}

n_cat = len(variables_cat)
fig, axes = plt.subplots(1, n_cat, figsize=(5 * n_cat, 6))
if n_cat == 1:
    axes = [axes]

for ax, (var, etiquetas) in zip(axes, variables_cat.items()):
    if var not in df.columns:
        ax.set_visible(False)
        continue

    columna_mapeada = df[var].map(etiquetas)
    tabla = df.groupby([df[COLUMNA_MMR], columna_mapeada]).size().unstack(fill_value=0)
    tabla_pct = tabla.div(tabla.sum(axis=1), axis=0) * 100

   
    if var in ['KRAS_STATUS', 'MLH1_PROMOTOR_METHYLATION_RESULTS']:
        tabla_pct = tabla_pct.iloc[:, ::-1]

    tabla_pct.plot(kind='bar', stacked=True, ax=ax,
               color=['#87cefa', '#cd0000', '#7a378b', '#6aaa64'][:tabla_pct.shape[1]],
               edgecolor='white', linewidth=0.8)

    ax.set_title(var, fontsize=11, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('% muestras')
    ax.tick_params(axis='x', rotation=30)
    
    
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], title=var, loc='upper right', fontsize=8)
              
    ax.set_ylim(0, 115)

plt.tight_layout()
plt.savefig(CARPETA_SALIDA + 'Barras_categoricas_MMR_invertidas.png', dpi=300, bbox_inches='tight')
plt.show()

# %%


import pandas as pd

df = pd.read_excel('/Users/andreasantiago/Desktop/tfm/datos/Datos_MSI_filtrado.xlsx')


COLUMNA_MMR = 'MMR_TYPE_IHQ' if 'MMR_TYPE_IHQ' in df.columns else 'MMR_proteina'

df = df[df[COLUMNA_MMR].notna() & (df[COLUMNA_MMR] != '')]
df[COLUMNA_MMR] = df[COLUMNA_MMR].astype(str).str.strip()

print("=== LYNCH_SYNDROME_DICHOTOMIC ===")
kras_totales = df['LYNCH_SYNDROME_DICHOTOMIC'].value_counts(dropna=False)
print(kras_totales)

print("\n LYNCH_SYNDROME_DICHOTOMIC ")

kras_mutados = df[df['LYNCH_SYNDROME_DICHOTOMIC'] == 1.0]


conteo_por_proteina = kras_mutados[COLUMNA_MMR].value_counts()
for prot, conteo in conteo_por_proteina.items():
    print(f"- {prot}: {conteo} muestras mutadas")

print(f"\nTotal de LYNCH_SYNDROME_DICHOTOMIC en todas las proteínas: {len(kras_mutados)}")