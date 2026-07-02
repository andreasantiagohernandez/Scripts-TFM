#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 11:29:45 2026

@author: andreasantiago
"""

import pandas as pd
import re

# =====================================================================
# 1. RUTAS A LOS ARCHIVOS
# =====================================================================

ruta_archivo = "/Users/andreasantiago/Desktop/tfm/datos/multiqc/multiqc_files_report_data/mqc_fastq_screen_plot_1.txt"

ruta_salida = "/Users/andreasantiago/Desktop/tfm/datos/multiqc/multiqc_files_report_data/muestras_seleccionadas_human_exclusivo.csv"

# %%

# =====================================================================
# 2. PROCESAMIENTO DE LOS DATOS
# =====================================================================
df_plot = pd.read_csv(ruta_archivo, sep="\t")

# Calcular el total de lecturas sumando todas las columnas numéricas 
df_plot['Total_Reads'] = df_plot.iloc[:, 1:].sum(axis=1)

# Calcular el porcentaje exclusivo de humano (la barra azul claro)
df_plot['Human_Pct_Exclusive'] = (df_plot['Human'] / df_plot['Total_Reads']) * 100

# Extraer "MSI_XXX" a partir de nombres como "MSI_101_1_fq1_fastq_screen"
df_plot['Base_Sample'] = df_plot['Sample'].str.extract(r'(MSI_\d+)')

# Agrupar por paciente/muestra base y calcular la media de todas sus réplicas
df_grouped = df_plot.groupby('Base_Sample')['Human_Pct_Exclusive'].mean().reset_index()

# Ordenar de mayor a menor porcentaje para que las mejores queden arriba
df_grouped = df_grouped.sort_values(by='Human_Pct_Exclusive', ascending=False)

# Dividir el dataframe en las que superan y las que no superan el umbral del 30%
superan_30 = df_grouped[df_grouped['Human_Pct_Exclusive'] > 30]
no_superan_30 = df_grouped[df_grouped['Human_Pct_Exclusive'] <= 30]



# %%

# =====================================================================
# 3. MOSTRAR INFORMACIÓN RELEVANTE POR PANTALLA
# =====================================================================
print("-" * 60)
print(" RESUMEN DEL FILTRADO DE MUESTRAS (HUMANO EXCLUSIVO)")
print("-" * 60)
print(f"Total de muestras únicas evaluadas: {len(df_grouped)}")
print(f"Porcentaje de mapeo medio global:   {df_grouped['Human_Pct_Exclusive'].mean():.2f}%\n")

print(f" MUESTRAS QUE SUPERAN EL 30% ({len(superan_30)} muestras):")
print(f"   - Media de mapeo de este grupo: {superan_30['Human_Pct_Exclusive'].mean():.2f}%")
print(f"   - Mejor muestra del proyecto:   {superan_30.iloc[0]['Base_Sample']} ({superan_30.iloc[0]['Human_Pct_Exclusive']:.2f}%)")
print("   - Lista de IDs para la matriz de conteos:")
print("     " + ", ".join(superan_30['Base_Sample'].tolist()))

print("\n" + "-" * 60)

print(f" MUESTRAS DESCARTADAS (<= 30%) ({len(no_superan_30)} muestras):")
print(f"   - Media de mapeo de este grupo: {no_superan_30['Human_Pct_Exclusive'].mean():.2f}%")
print(f"   - Peor muestra del proyecto:    {no_superan_30.iloc[-1]['Base_Sample']} ({no_superan_30.iloc[-1]['Human_Pct_Exclusive']:.2f}%)")
print("   - Lista de IDs a excluir del análisis:")
print("     " + ", ".join(no_superan_30['Base_Sample'].tolist()))

print("-" * 60 + "\n")

# %%

# =====================================================================
# 4. GUARDAR RESULTADOS
# =====================================================================
superan_30.to_csv(ruta_salida, index=False)
print(f" Archivo CSV con las {len(superan_30)} muestras aprobadas guardado en:\n   {ruta_salida}")