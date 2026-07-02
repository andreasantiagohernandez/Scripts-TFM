#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 25 10:09:11 2026

@author: andreasantiago
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import gseapy as gp
from gseapy import dotplot
import matplotlib.pyplot as plt
from pathlib import Path
from loguru import logger

# =============================================================================
# 1. CONFIGURACIÓN
# =============================================================================

DEG_dir = Path.home() / "/Users/andreasantiago/Desktop/tfm/datos/Resultados_DESeq2_MMR_medioestrictos"

output_dir = Path.home() / "Desktop/tfm/datos/gsea/resultados_GSEA"
output_dir.mkdir(parents=True, exist_ok=True)

logger.add(output_dir / "gsea.log", level="DEBUG")

GRUPOS = {
    "MLH1_PMS2":      "Todos_genes_MLH1_PMS2_vs_Resto.csv",
    "MSH2_MSH6":      "Todos_genes_MSH2_MSH6_vs_Resto.csv",
    "MSH6":           "Todos_genes_MSH6_vs_Resto.csv",
    "MSH2":           "Todos_genes_MSH2_vs_Resto.csv",
    "PMS2":           "Todos_genes_PMS2_vs_Resto.csv",
    "MLH1":           "Todos_genes_MLH1_vs_Resto.csv",
    "MSH6_PMS2_MLH1": "Todos_genes_MSH6_PMS2_MLH1_vs_Resto.csv"
}

# Gene set:
GENE_SETS = [
    "MSigDB_Hallmark_2020"
]

# %%

# =============================================================================
# 2. FUNCIONES
# =============================================================================

def build_rnk(df: pd.DataFrame) -> pd.Series:
    """
    Construye el ranking a partir del output COMPLETO de PyDESeq2.
    Usa Gene_Symbol porque MSigDB/Enrichr trabaja con símbolos génicos.
    """
    df = df.copy()
    df = df.rename(columns={
        "Gene_ID": "gene_id",
        "Gene_Symbol": "gene_symbol"
    })

    if "gene_symbol" not in df.columns:
        raise ValueError("No se encontró la columna Gene_Symbol en el archivo.")

    df["gene_symbol"] = df["gene_symbol"].astype(str).str.upper().str.strip()

    df = df[df["gene_symbol"].notna()]
    df = df[df["gene_symbol"] != ""]
    df = df[df["gene_symbol"] != "NAN"]
    df = df[df["gene_symbol"] != "NA"]
    df["padj"] = df["padj"].replace(0, 1e-300).fillna(1)
    df["log2FoldChange"] = df["log2FoldChange"].fillna(0)
    df["Rank"] = (
        -np.log10(df["padj"]) * np.sign(df["log2FoldChange"])
        + df["log2FoldChange"] * 0.01
    )

    df = df.sort_values("Rank", ascending=False).drop_duplicates("gene_symbol")

    rnk = df.set_index("gene_symbol")["Rank"].sort_values(ascending=False)

    logger.info(f"  Ranking: {len(rnk)} genes | max={rnk.max():.2f} | min={rnk.min():.2f}")
    return rnk


def run_gsea(rnk: pd.Series, grupo: str, gene_set: str, output_base: Path):
    outdir = output_base / grupo / gene_set.replace(" ", "_")
    outdir.mkdir(parents=True, exist_ok=True)

    logger.info(f"  [GSEA] {grupo} | {gene_set} | genes en ranking: {len(rnk)}")

    try:
        pre_res = gp.prerank(
            rnk=rnk,
            gene_sets=gene_set,
            threads=4,
            min_size=5,
            max_size=500,
            permutation_num=1000,
            outdir=str(outdir),
            format='png',
            seed=42,
            verbose=False,
        )
    except Exception as e:
        logger.error(f"  [GSEA] Error en prerank {grupo}/{gene_set}: {e}")
        return None, None

    if pre_res is None or pre_res.res2d.empty:
        logger.warning(f"  [GSEA] Sin resultados para {grupo}/{gene_set}")
        return None, None

    res = pre_res.res2d.copy()

    csv_path = outdir / "gsea_results_completo.csv"
    res.to_csv(csv_path, index=False)
    logger.success(f"  [GSEA] CSV guardado: {csv_path}")

    sig = res[res["FDR q-val"] < 0.25].sort_values("FDR q-val")
    logger.info(f"  [GSEA] Rutas con FDR<0.25: {len(sig)}")

    sig_csv = outdir / "gsea_results_significativos.csv"
    sig.to_csv(sig_csv, index=False)

    if not res.empty:
        try:
            dotplot(
                pre_res.res2d,
                column="FDR q-val",
                title=f"GSEA {grupo}\n{gene_set}",
                cmap=plt.cm.RdYlBu_r,
                size=10,
                cutoff=0.25,
                top_term=15,
                ofname=str(outdir / "gsea_dotplot.png"),
            )
            plt.close("all")
            logger.success("  [GSEA] Dotplot guardado")
        except Exception as e:
            logger.warning(f"  [GSEA] Error en dotplot: {e}")

    if not sig.empty:
        top_terms = sig["Term"].head(5).tolist()
        try:
            pre_res.plot(
                terms=top_terms,
                show_ranking=True,
                legend_kws={"loc": "upper left"},
            )
            plt.suptitle(f"{grupo} | {gene_set}", fontsize=9)
            plt.savefig(outdir / "enrichment_plots_top5.png", dpi=150, bbox_inches="tight")
            plt.close("all")
            logger.success(f"  [GSEA] Enrichment plots guardados: {top_terms}")
        except Exception as e:
            logger.warning(f"  [GSEA] Error en enrichment plots: {e}")

    return pre_res, sig


def summarize_all(all_results: dict, output_base: Path) -> pd.DataFrame:
    rows = []
    for grupo, gs_dict in all_results.items():
        for gene_set, (_, sig) in gs_dict.items():
            if sig is not None and not sig.empty:
                tmp = sig.copy()
                tmp["Grupo"] = grupo
                tmp["GeneSet_DB"] = gene_set
                rows.append(tmp)

    if rows:
        summary = pd.concat(rows, ignore_index=True)
        summary_path = output_base / "RESUMEN_rutas_significativas.csv"
        summary.to_csv(summary_path, index=False)
        logger.success(f"Resumen guardado: {summary_path}")
        return summary

    logger.warning("No se encontraron rutas significativas en ningún grupo.")
    return pd.DataFrame()

# %%


# =============================================================================
# 3. EJECUCIÓN PRINCIPAL
# =============================================================================

all_results = {}

for grupo, archivo in GRUPOS.items():
    filepath = DEG_dir / archivo

    if not filepath.exists():
        logger.error(f"Archivo no encontrado: {filepath} — saltando {grupo}")
        continue

    logger.info(f"========== {grupo} ==========")

    df = pd.read_csv(filepath)
    logger.info(f"  Genes cargados: {len(df)} | columnas: {list(df.columns)}")

    df_cols = set(df.columns)
    required_cols = {"Gene_Symbol", "log2FoldChange", "padj"}
    missing = required_cols - df_cols
    if missing:
        logger.error(f"  Faltan columnas en {archivo}: {missing}")
        continue

    try:
        rnk = build_rnk(df)
    except Exception as e:
        logger.error(f"  Error construyendo ranking para {grupo}: {e}")
        continue

    all_results[grupo] = {}
    for gs in GENE_SETS:
        pre_res, sig = run_gsea(rnk, grupo, gs, output_dir)
        if pre_res is not None:
            all_results[grupo][gs] = (pre_res, sig)

summary_df = summarize_all(all_results, output_dir)

print("\n" + "=" * 60)
if not summary_df.empty:
    print("✅  Análisis completado.")
    print(f"    Rutas significativas totales (FDR<0.25): {len(summary_df)}")
    cols_show = ["Grupo", "Term", "NES", "FDR q-val", "GeneSet_DB"]
    cols_show = [c for c in cols_show if c in summary_df.columns]
    print(summary_df[cols_show].head(20).to_string(index=False))
else:
    print("⚠️   No se encontraron rutas significativas.")
    print("    Revisa: Desktop/gsea/resultados_GSEA/gsea.log")
print("=" * 60)