#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 11:49:13 2026

@author: andreasantiago
"""

import pandas as pd
import glob
import os

# =====================================================================
# 1. CONFIGURACIÓN DE RUTAS
# =====================================================================
ruta_csv_muestras = "/Users/andreasantiago/Desktop/tfm/datos/multiqc/multiqc_files_report_data/muestras_seleccionadas_human_exclusivo.csv"

carpeta_htseq = "/Users/andreasantiago/Desktop/tfm/datos/quant/star/htseq/"

ruta_matriz_salida = "/Users/andreasantiago/Desktop/tfm/datos/matriz_conteos_filtrada.tsv"


# %%


# =====================================================================
# 2. CARGAR LA LISTA DE MUESTRAS VÁLIDAS
# =====================================================================
df_muestras = pd.read_csv(ruta_csv_muestras)
muestras_validas = set(df_muestras['Base_Sample'].tolist())

print(f"Buscando {len(muestras_validas)} muestras válidas...")


# %%


# =====================================================================
# 3. LEER Y UNIR LOS ARCHIVOS .TAB
# =====================================================================
archivos_tab = glob.glob(os.path.join(carpeta_htseq, "*.tab"))

matriz_final = pd.DataFrame()
archivos_procesados = 0

for archivo in archivos_tab:
    nombre_archivo = os.path.basename(archivo)
    import re
    match = re.search(r'(MSI_\d+)', nombre_archivo)
    
    if match:
        base_id = match.group(1)
        
        if base_id in muestras_validas:
            df_temp = pd.read_csv(archivo, sep='\t', header=None, names=['Gene_ID', base_id])
            
            if matriz_final.empty:
                matriz_final = df_temp
            else:
                matriz_final = pd.merge(matriz_final, df_temp, on='Gene_ID', how='outer')
            
            archivos_procesados += 1

# %%


# =====================================================================
# 4. LIMPIEZA FINAL Y EXPORTACIÓN
# =====================================================================
matriz_final = matriz_final[~matriz_final['Gene_ID'].str.startswith('__')]

matriz_final = matriz_final.sort_values('Gene_ID')

matriz_final.to_csv(ruta_matriz_salida, sep='\t', index=False)

print(f"\n¡Éxito! Se procesaron {archivos_procesados} archivos .tab.")
print(f"La matriz final tiene {len(matriz_final)} genes y {len(matriz_final.columns)-1} muestras.")
print(f"Guardada en: {ruta_matriz_salida}")


