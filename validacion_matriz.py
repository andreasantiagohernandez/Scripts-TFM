#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 12:31:45 2026

@author: andreasantiago
"""

import pandas as pd
import numpy as np

# =====================================================================
# SCRIPT DE VALIDACIÓN DE MATRIZ DE CONTEOS (PYTHON)
# =====================================================================

ruta_matriz = "/Users/andreasantiago/Desktop/tfm/datos/matriz_conteos_filtrada.tsv"

# %%

df = pd.read_csv(ruta_matriz, sep="\t", index_col=0)

print("\n===================================================")
print("REPORTE DE VALIDACIÓN DE LA MATRIZ DE CONTEOS")
print("===================================================\n")

# --- PRUEBA 1: Dimensiones ---
n_genes, n_muestras = df.shape
print("1. DIMENSIONES:")
print(f"   - Genes detectados (filas): {n_genes}")
print(f"   - Muestras (columnas): {n_muestras}")
if n_muestras == 168:
    print("   CORRECTO: Tienes exactamente las 168 muestras esperadas.")
else:
    print(f"   ERROR: Se esperaban 168 muestras, pero hay {n_muestras}.")


# %%


# --- PRUEBA 2: Valores Nulos (NA/NaN) ---
print("\n2. VALORES NULOS (NA/NaN):")
n_nas = df.isna().sum().sum()
if n_nas == 0:
    print("   CORRECTO: No hay valores vacíos en la matriz.")
else:
    print(f"  ERROR: Se detectaron {n_nas} valores NaN. DESeq2 fallará en R.")
# %%


# --- PRUEBA 3: Valores Negativos o Decimales ---
print("\n3. TIPO DE DATOS (Enteros positivos):")
try:
    hay_negativos = (df < 0).any().any()
    hay_decimales = (df.fillna(0) % 1 != 0).any().any()
    
    if not hay_negativos and not hay_decimales:
        print("   CORRECTO: Todos los valores son números enteros positivos (conteos puros).")
    else:
        print("   ERROR: La matriz contiene decimales o números negativos.")
except TypeError:
    print("   ERROR: La matriz contiene texto donde debería haber números.")
    hay_negativos = True # Para forzar el fallo en el resultado final
# %%


# --- PRUEBA 4: Basura de HTSeq ---
print("\n4. METADATOS DE HTSeq (__no_feature, etc):")
basura_detectada = df.index.str.startswith('__').any()
if not basura_detectada:
    print("   CORRECTO: Las filas estadísticas de HTSeq fueron eliminadas correctamente.")
else:
    print("   ERROR: Quedan genes que empiezan por '__'. Debes borrarlos.")
# %%


# --- PRUEBA 5: Sanity Check (Conteo Total) ---
print("\n5. MUESTRA ALEATORIA (Sanity Check):")
muestra_azar = np.random.choice(df.columns)
total_lecturas = df[muestra_azar].sum()

print(f"   - La muestra '{muestra_azar}' tiene un total de {total_lecturas:,.0f} lecturas mapeadas a genes.")
print(f"   Abre el archivo .tab original de '{muestra_azar}' y suma sus lecturas.")
print(f"   Debe dar exactamente {total_lecturas:,.0f} para confirmar que el merge fue perfecto.")

print("\n===================================================")
if n_muestras == 168 and n_nas == 0 and not hay_negativos and not hay_decimales and not basura_detectada:
    print( "LA MATRIZ ES PERFECTA. ¡Lista para exportary usar en DESeq2!")
else:
    print("⚠️ RESULTADO: La matriz tiene errores. Revisa los puntos con ❌ antes de seguir.")
print("===================================================\n")