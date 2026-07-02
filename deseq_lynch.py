#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 14:10:07 2026

@author: andreasantiago
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 13:31:13 2026

@author: andreasantiago
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 12:49:08 2026

@author: andreasantiago
"""

### MLH1 VS MLH1/PMS2



# Minimal edit: only change the comparison focus to MLH1 vs MLH1/PMS2.
# Everything else is kept as close as possible to your script.

import pandas as pd
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
from pydeseq2.default_inference import DefaultInference
import os

# =====================================================================
# 1. CARGA DE DATOS Y FILTRADO CLÍNICO
# =====================================================================
print("Cargando datos...")
counts = pd.read_csv("matriz_conteos_filtrada.tsv", sep="\t", index_col=0).T
metadata = pd.read_csv("/Users/andreasantiago/Desktop/tfm/datos/LYNCH/grupos_Lynch.csv")

COLUMNA_MMR = 'LYNCH_SYNDROME_DICHOTOMIC' if 'LYNCH_SYNDROME_DICHOTOMIC' in metadata.columns else 'LYNCH_SYNDROME_DICHOTOMIC'
metadata = metadata[metadata[COLUMNA_MMR].notna() & (metadata[COLUMNA_MMR] != '')]
metadata['CODE'] = metadata['CODE'].astype(str).str.replace(' ', '_').str.strip()
metadata.set_index('CODE', inplace=True)

mapeo_mmr = {1: 'MLH1', 2: 'MSH6', 3: 'PMS2', 4: 'MLH1/PMS2', 6: 'MSH2/MSH6', 9: 'MSH6/PMS2/MLH1', 10: 'Otros'}
metadata['LYNCH_SYNDROME_DICHOTOMIC'] = metadata[COLUMNA_MMR].replace(mapeo_mmr).astype(str)

muestras_comunes = counts.index.intersection(metadata.index)
counts = counts.loc[muestras_comunes]
metadata = metadata.loc[muestras_comunes]

assert all(counts.index == metadata.index), "Error: Nombres no coinciden."
# %%


# =====================================================================
# 2. FILTRADO ESTRICTO DE GENES (Mejora la potencia estadística)
# =====================================================================
print("Filtrando genes con baja expresión...")
genes_validos = counts.columns[(counts >= 10).sum(axis=0) >= 5]
counts = counts[genes_validos]
print(f"Genes conservados para el análisis: {len(genes_validos)}")

# =====================================================================
# 3. CARGAR BASE DE DATOS HGNC
# =====================================================================
CARPETA_SALIDA = "LYNCH"
os.makedirs(CARPETA_SALIDA, exist_ok=True)

diccionario_genes = {}
try:
    hgnc = pd.read_csv("hgnc_complete_set.txt", sep="\t", usecols=["ensembl_gene_id", "symbol"], low_memory=False)
    hgnc.dropna(subset=["ensembl_gene_id"], inplace=True)
    hgnc.set_index("ensembl_gene_id", inplace=True)
    diccionario_genes = hgnc['symbol'].to_dict()
    print("Base de datos HGNC cargada correctamente.")
except FileNotFoundError:
    print("⚠️ No se encontró 'hgnc_complete_set.txt'. Se usarán los IDs de Ensembl.")
# %%


# =====================================================================
# 4. ANÁLISIS DESeq2 MLH1 vs MLH1/PMS2
# =====================================================================
resumen_total = []
grupo_ref = 'SI'
grupo_problema = 'NO'
if grupo_problema not in metadata['LYNCH_SYNDROME_DICHOTOMIC'].unique():
    raise ValueError(f"No existe el grupo {grupo_problema} en metadata['LYNCH_SYNDROME_DICHOTOMIC']")
if grupo_ref not in metadata['LYNCH_SYNDROME_DICHOTOMIC'].unique():
    raise ValueError(f"No existe el grupo {grupo_ref} en metadata['LYNCH_SYNDROME_DICHOTOMIC']")

nombre_seguro = f"{grupo_problema}_vs_{grupo_ref}".replace('/', '_')
print(f"\n--- Analizando: {grupo_problema} vs {grupo_ref} ---")

metadata['Comparacion_OvR'] = [grupo_problema if g == grupo_problema else grupo_ref for g in metadata['LYNCH_SYNDROME_DICHOTOMIC']]
metadata = metadata[metadata['Comparacion_OvR'].isin([grupo_problema, grupo_ref])].copy()

counts = counts.loc[metadata.index]

# Por seguridad, DESeq2 necesita dos grupos reales
if metadata['Comparacion_OvR'].nunique() < 2:
    raise ValueError("No hay dos grupos distintos para comparar.")

dds = DeseqDataSet(
    counts=counts,
    metadata=metadata,
    design_factors="Comparacion_OvR",
    refit_cooks=True,
    inference=DefaultInference(n_cpus=4)
)

try:
    dds.deseq2()
    ds = DeseqStats(
        dds,
        contrast=["Comparacion_OvR", grupo_problema, grupo_ref],
        inference=DefaultInference(n_cpus=4)
    )
    ds.summary()
    resultados = ds.results_df

    ids_limpios = resultados.index.str.split('.').str[0]
    nombres_traducidos = pd.Series(ids_limpios).map(diccionario_genes).fillna(pd.Series(resultados.index.values))
    resultados.insert(0, 'Gene_Symbol', nombres_traducidos.values)

    res_validos = resultados.dropna(subset=['padj'])
    sobre = res_validos[(res_validos['padj'] < 0.05) & (res_validos['log2FoldChange'] > 1)]
    infra = res_validos[(res_validos['padj'] < 0.05) & (res_validos['log2FoldChange'] < -1)]

    top_up = sobre.sort_values('log2FoldChange', ascending=False).head(5)['Gene_Symbol'].tolist()
    top_down = infra.sort_values('log2FoldChange', ascending=True).head(5)['Gene_Symbol'].tolist()

    resultados.to_csv(f"{CARPETA_SALIDA}/Todos_genes_{nombre_seguro}.csv")
    pd.concat([sobre, infra]).sort_values("padj").to_csv(f"{CARPETA_SALIDA}/Significativos_{nombre_seguro}.csv")

    resumen_total.append({
        "Proteína Evaluada": grupo_problema,
        "Referencia": grupo_ref,
        "Sobreexpresados": len(sobre),
        "Infraexpresados": len(infra),
        "Total DEGs": len(sobre) + len(infra),
        "Top 5 Sobreexpresados": ", ".join(top_up) if top_up else "Ninguno",
        "Top 5 Infraexpresados": ", ".join(top_down) if top_down else "Ninguno"
    })

except Exception as e:
    print(f"❌ Error procesando {grupo_problema}: {e}")

# =====================================================================
# 5. GUARDAR Y MOSTRAR RESUMEN FINAL
# =====================================================================
if resumen_total:
    df_resumen = pd.DataFrame(resumen_total)
    df_resumen.to_csv(f"{CARPETA_SALIDA}/Resumen_Final_MLH1PMS2_vs_MLH1.csv", index=False)
    print(f"\n✅ ANÁLISIS FINALIZADO. Archivos en la carpeta: '{CARPETA_SALIDA}'\n")
    print(df_resumen[['Proteína Evaluada', 'Referencia', 'Total DEGs']])

# =====================================================================
# RESUMEN ANTES DEL FILTRADO FINAL
# =====================================================================
res_validos = resultados.dropna(subset=['padj']).copy()

n_padj = (res_validos['padj'] < 0.05).sum()
n_lfc = (res_validos['log2FoldChange'].abs() > 1).sum()
n_ambos = ((res_validos['padj'] < 0.05) & (res_validos['log2FoldChange'].abs() > 1)).sum()

print(f"\nResumen previo para {grupo_problema}:")
print(f"Genes con padj < 0.05: {n_padj}")
print(f"Genes con |log2FC| > 1: {n_lfc}")
print(f"Genes que cumplen ambos: {n_ambos}")

efecto_grande_no_sig = res_validos[
    (res_validos['padj'] >= 0.05) & (res_validos['log2FoldChange'].abs() > 1)
].sort_values('log2FoldChange', key=lambda s: s.abs(), ascending=False)

print("Top genes con |log2FC| > 1 pero no significativos:")
print(efecto_grande_no_sig[['Gene_Symbol', 'log2FoldChange', 'padj']].head(10).to_string(index=False))

resumen_prev = pd.DataFrame([{
    'Grupo': grupo_problema,
    'Referencia': grupo_ref,
    'padj<0.05': n_padj,
    '|log2FC|>1': n_lfc,
    'Ambos': n_ambos,
    'Efecto grande no sig': len(efecto_grande_no_sig)
}])

resumen_prev.to_csv(f"{CARPETA_SALIDA}/Resumen_previo_{nombre_seguro}.csv", index=False)

# %%