# comparar_mpi_spark.py
#
# Lê os CSVs do MPI e do Spark e gera gráficos de tempo, speedup e eficiência.
#
# Uso:
#   python comparar_mpi_spark.py --mpi mpi_1.csv --spark spark_1.csv
#   python comparar_mpi_spark.py --mpi mpi_1.csv --spark spark_1.csv --saida graficos --sem-janela
#
# Definições, para cada algoritmo e N (T = ms_total, a tarefa/processo mais lento):
#   speedup próprio     S(p)  = T_fw(1) / T_fw(p)       baseline: o mesmo framework com p=1
#   speedup vs MPI      S(p)  = T_mpi(1) / T_fw(p)      baseline: MPI sequencial (p=1), comum aos dois
#   eficiência          E(p)  = S(p) / p
#
# Por que dois baselines: o speedup "próprio" mostra como cada framework escala, mas esconde
# que o Spark com p=1 pode ser mais lento/rápido que o MPI com p=1. O speedup "vs MPI"
# usa o mesmo ponto de partida para os dois e mostra quem realmente termina antes.
#
# Execuções repetidas da mesma configuração viram média, com barra de erro (desvio padrão)
# no gráfico de tempo. Configurações com uma única execução ficam sem barra.

import argparse
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

ORDEM_ALGOS = ["trivial", "raiz", "crivo"]
COR = {"MPI": "tab:blue", "Spark": "tab:orange"}
MARCA = {"MPI": "o", "Spark": "s"}


# ----------------------------------------------------------------------
# Leitura e cálculo
# ----------------------------------------------------------------------

def carregar(caminho_mpi, caminho_spark):
    """Une os dois CSVs: framework, algo, N, p, ms_total, ms_parede (NaN no MPI)."""
    mpi = pd.read_csv(caminho_mpi).rename(columns={"N_processos": "p"})
    spark = pd.read_csv(caminho_spark).rename(columns={"N_particoes": "p"})

    for df in (mpi, spark):
        if "ms_parede" not in df.columns:
            df["ms_parede"] = float("nan")

    colunas = ["framework", "algo", "N", "p", "ms_total", "ms_parede"]
    df = pd.concat([mpi[colunas], spark[colunas]], ignore_index=True)
    df["algo"] = df["algo"].str.strip().str.lower()
    df["framework"] = df["framework"].str.strip()
    return df


def agregar(df, estatistica):
    """Colapsa repetições: valor central (média/mediana/mín), desvio padrão e nº de execuções."""
    chaves = ["framework", "algo", "N", "p"]
    agg = df.groupby(chaves, as_index=False).agg(
        ms_total=("ms_total", estatistica),
        desvio=("ms_total", "std"),
        ms_parede=("ms_parede", estatistica),
        execucoes=("ms_total", "size"),
    )
    agg["desvio"] = agg["desvio"].fillna(0.0)  # 1 execução => sem desvio
    return agg


def calcular_metricas(agg):
    """Adiciona speedup/eficiência próprios e relativos ao MPI com p=1."""
    partes = []
    for (fw, algo, n), g in agg.groupby(["framework", "algo", "N"]):
        g = g.sort_values("p").copy()

        # baseline próprio: p=1 do mesmo framework (ou menor p, com aviso)
        p_ref = g["p"].min()
        if p_ref != 1:
            print(f"[aviso] {fw}/{algo}/N={n}: sem p=1; speedup próprio relativo a p={p_ref}.",
                  file=sys.stderr)
        t_ref = g.loc[g["p"] == p_ref, "ms_total"].iloc[0]
        g["p_ref"] = p_ref
        g["speedup"] = t_ref / g["ms_total"]
        g["eficiencia"] = g["speedup"] * p_ref / g["p"]

        # baseline comum: MPI com p=1
        base = agg[(agg["framework"] == "MPI") & (agg["algo"] == algo)
                   & (agg["N"] == n) & (agg["p"] == 1)]
        if base.empty:
            g["speedup_vs_mpi"] = float("nan")
            g["eficiencia_vs_mpi"] = float("nan")
        else:
            g["speedup_vs_mpi"] = base["ms_total"].iloc[0] / g["ms_total"]
            g["eficiencia_vs_mpi"] = g["speedup_vs_mpi"] / g["p"]
        partes.append(g)

    return pd.concat(partes, ignore_index=True)


# ----------------------------------------------------------------------
# Gráficos
# ----------------------------------------------------------------------

def _grade(res, titulo):
    """Uma linha por N, uma coluna por algoritmo."""
    ns = sorted(res["N"].unique())
    algos = [a for a in ORDEM_ALGOS if a in set(res["algo"])]
    fig, eixos = plt.subplots(len(ns), len(algos), squeeze=False,
                              figsize=(5.2 * len(algos), 4.0 * len(ns)),
                              constrained_layout=True)
    fig.suptitle(titulo, fontsize=14)
    return fig, eixos, ns, algos


def _acabamento(ax, n, algo, procs, ylabel, primeira_coluna):
    ax.set_title(f"{algo}  |  N = {n:,}".replace(",", "."), fontsize=10)
    ax.set_xlabel("processos (MPI) / partições (Spark)")
    if primeira_coluna:
        ax.set_ylabel(ylabel)
    ax.set_xticks(procs)
    ax.set_xticklabels([str(p) for p in procs])
    ax.grid(True, alpha=0.3)


def _desenhar(ax, d, y, rotulo, fw, ls="-", marker=None):
    """Linha com marcador; um ponto isolado (ex.: MPI só com p=1) fica visível do mesmo jeito."""
    ax.plot(d["p"], d[y], label=rotulo, color=COR[fw], ls=ls,
            marker=marker or MARCA[fw], markersize=7)


def grafico_tempo(res):
    fig, eixos, ns, algos = _grade(res, "Tempo de execução: MPI vs Spark")

    for i, n in enumerate(ns):
        for j, algo in enumerate(algos):
            ax = eixos[i][j]
            sub = res[(res["N"] == n) & (res["algo"] == algo)]
            procs = sorted(sub["p"].unique())

            mpi = sub[sub["framework"] == "MPI"].sort_values("p")
            spark = sub[sub["framework"] == "Spark"].sort_values("p")

            if not mpi.empty:
                rot = "MPI (média ± desvio)" if mpi["execucoes"].max() > 1 else "MPI"
                ax.errorbar(mpi["p"], mpi["ms_total"], yerr=mpi["desvio"], color=COR["MPI"],
                            marker=MARCA["MPI"], markersize=8, capsize=4, ls="-", label=rot)
                # referência horizontal: tempo sequencial do MPI
                base = mpi.loc[mpi["p"] == 1, "ms_total"]
                if not base.empty and len(mpi) == 1:
                    ax.axhline(base.iloc[0], color=COR["MPI"], ls=":", alpha=0.6)

            if not spark.empty:
                _desenhar(ax, spark, "ms_total", "Spark (tarefa mais lenta)", "Spark")
                if spark["ms_parede"].notna().any():
                    _desenhar(ax, spark, "ms_parede", "Spark (parede, com overhead)", "Spark",
                              ls="--", marker="^")

            ax.set_yscale("log")
            _acabamento(ax, n, algo, procs, "tempo (ms, escala log)", j == 0)
            if i == 0 and j == 0:
                ax.legend(fontsize=8)
    return fig


def _grafico_metrica(res, titulo, col_propria, col_vs_mpi, ylabel, ideal):
    fig, eixos, ns, algos = _grade(res, titulo)

    for i, n in enumerate(ns):
        for j, algo in enumerate(algos):
            ax = eixos[i][j]
            sub = res[(res["N"] == n) & (res["algo"] == algo)]
            procs = sorted(sub["p"].unique())

            ideal(ax, procs)
            for fw in ["MPI", "Spark"]:
                d = sub[sub["framework"] == fw].sort_values("p")
                if d.empty:
                    continue
                if len(d) > 1:  # com um ponto só, o speedup próprio é trivialmente 1
                    _desenhar(ax, d, col_propria, f"{fw} (baseline: {fw} p=1)", fw)
                if fw != "MPI" and d[col_vs_mpi].notna().any():
                    _desenhar(ax, d, col_vs_mpi, f"{fw} (baseline: MPI p=1)", fw,
                              ls="--", marker="^")

            # MPI sequencial: ponto de partida comum
            mpi1 = sub[(sub["framework"] == "MPI") & (sub["p"] == 1)]
            if not mpi1.empty:
                ax.plot([1], [1], marker="o", color=COR["MPI"], ls="", markersize=8,
                        label="MPI p=1 (baseline)")

            _acabamento(ax, n, algo, procs, ylabel, j == 0)
            if i == 0 and j == 0:
                ax.legend(fontsize=8)
    return fig


def _ideal_speedup(ax, procs):
    ax.plot(procs, procs, color="gray", ls=":", label="ideal (linear)")


def _ideal_eficiencia(ax, procs):
    ax.axhline(1.0, color="gray", ls=":", label="ideal (100%)")


def grafico_speedup(res):
    return _grafico_metrica(res, "Speedup: MPI vs Spark", "speedup", "speedup_vs_mpi",
                            "speedup", _ideal_speedup)


def grafico_eficiencia(res):
    fig = _grafico_metrica(res, "Eficiência: MPI vs Spark", "eficiencia", "eficiencia_vs_mpi",
                           "eficiência (speedup / p)", _ideal_eficiencia)
    for ax in fig.axes:
        ax.set_ylim(bottom=0)
    return fig


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Compara MPI e Spark: tempo, speedup e eficiência.")
    ap.add_argument("--mpi", default="mpi.csv")
    ap.add_argument("--spark", default="spark.csv")
    ap.add_argument("--saida", default="graficos", help="pasta onde salvar PNGs e resumo.csv")
    ap.add_argument("--estatistica", choices=["mean", "median", "min"], default="mean",
                    help="como combinar execuções repetidas (padrão: média)")
    ap.add_argument("--sem-janela", action="store_true", help="só salva os PNGs")
    args = ap.parse_args()

    df = carregar(args.mpi, args.spark)
    res = calcular_metricas(agregar(df, args.estatistica))

    os.makedirs(args.saida, exist_ok=True)
    res.to_csv(os.path.join(args.saida, "resumo.csv"), index=False)
    print(res.round(3).to_string(index=False))

    for nome, fig in {"tempo": grafico_tempo(res),
                      "speedup": grafico_speedup(res),
                      "eficiencia": grafico_eficiencia(res)}.items():
        caminho = os.path.join(args.saida, f"{nome}.png")
        fig.savefig(caminho, dpi=150)
        print(f"salvo: {caminho}")

    if not args.sem_janela:
        plt.show()


if __name__ == "__main__":
    main()