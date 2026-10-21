# comparar_mpi_spark.py
#
# Lê mpi.csv e spark.csv e compara os dois frameworks em gráficos (matplotlib):
#   tempo.png            tempo de execução (ms) por nº de processos/partições
#   speedup.png          speedup de cada framework
#   eficiencia.png       eficiência de cada framework
#   comparacao_direta.png  razão MPI/Spark no mesmo p (>1 = Spark mais rápido)
#
# Uso:
#   python comparar_mpi_spark.py
#   python comparar_mpi_spark.py --mpi mpi.csv --spark spark.csv --saida graficos --sem-janela
#
# Duas métricas de tempo, nos dois frameworks:
#   ms_total   processo/tarefa MAIS LENTO, só computação
#   ms_parede  ponta a ponta (MPI: até o rank 0 ter todos os primos; Spark: collect() no driver)
#
# Definições (cada framework contra ele mesmo, com a mesma métrica):
#   speedup(p)    = T(1) / T(p)
#   eficiência(p) = speedup(p) / p
# Se faltar p=1 para alguma configuração, usa o menor p disponível (e avisa).
#
# Execuções repetidas da mesma configuração viram média; se houver repetições,
# o gráfico de tempo mostra barras de erro (desvio padrão).

import argparse
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

ORDEM_ALGOS = ["trivial", "raiz", "crivo"]
COR = {"MPI": "tab:blue", "Spark": "tab:orange"}
MARCA = {"MPI": "o", "Spark": "s"}
METRICAS = {"total": "ms_total", "parede": "ms_parede"}
ESTILO_METRICA = {"total": {"ls": "-", "alpha": 1.0}, "parede": {"ls": "--", "alpha": 0.75}}
ROTULO_METRICA = {"total": "ms_total", "parede": "ms_parede"}


# ----------------------------------------------------------------------
# Leitura e cálculo
# ----------------------------------------------------------------------

def carregar(caminho_mpi, caminho_spark):
    """Une os dois CSVs: framework, algo, N, p, ms_total, ms_parede."""
    mpi = pd.read_csv(caminho_mpi).rename(columns={"N_processos": "p"})
    spark = pd.read_csv(caminho_spark).rename(columns={"N_particoes": "p"})

    for df in (mpi, spark):  # CSV antigo sem ms_parede continua funcionando
        if "ms_parede" not in df.columns:
            df["ms_parede"] = float("nan")

    colunas = ["framework", "algo", "N", "p", "ms_total", "ms_parede"]
    df = pd.concat([mpi[colunas], spark[colunas]], ignore_index=True)
    df["algo"] = df["algo"].str.strip().str.lower()
    df["framework"] = df["framework"].str.strip()
    return df


def agregar(df, estatistica):
    """Colapsa repetições: valor central, desvio padrão e nº de execuções."""
    agg = df.groupby(["framework", "algo", "N", "p"], as_index=False).agg(
        ms_total=("ms_total", estatistica),
        ms_parede=("ms_parede", estatistica),
        desvio_total=("ms_total", "std"),
        desvio_parede=("ms_parede", "std"),
        execucoes=("ms_total", "size"),
    )
    return agg.fillna({"desvio_total": 0.0, "desvio_parede": 0.0})


def calcular_metricas(agg):
    """Speedup e eficiência de cada framework contra ele mesmo, para ms_total e ms_parede."""
    partes = []
    for (fw, algo, n), g in agg.groupby(["framework", "algo", "N"]):
        g = g.sort_values("p").copy()
        p_ref = g["p"].min()
        if p_ref != 1:
            print(f"[aviso] {fw}/{algo}/N={n}: sem p=1; speedup relativo a p={p_ref}.",
                  file=sys.stderr)
        g["p_ref"] = p_ref

        for chave, coluna in METRICAS.items():
            t_ref = g.loc[g["p"] == p_ref, coluna].iloc[0]
            g[f"speedup_{chave}"] = t_ref / g[coluna]
            g[f"eficiencia_{chave}"] = g[f"speedup_{chave}"] * p_ref / g["p"]
        partes.append(g)

    return pd.concat(partes, ignore_index=True)


def calcular_razao(res):
    """Razão MPI/Spark no mesmo p. Valores > 1 significam que o Spark foi mais rápido."""
    mpi = res[res["framework"] == "MPI"].set_index(["algo", "N", "p"])
    spark = res[res["framework"] == "Spark"].set_index(["algo", "N", "p"])
    comuns = mpi.index.intersection(spark.index)
    if len(comuns) == 0:
        return pd.DataFrame()

    out = pd.DataFrame(index=comuns)
    for chave, coluna in METRICAS.items():
        out[f"razao_{chave}"] = mpi.loc[comuns, coluna] / spark.loc[comuns, coluna]
    return out.reset_index()


# ----------------------------------------------------------------------
# Gráficos
# ----------------------------------------------------------------------

def _grade(res, titulo):
    """Uma linha por N, uma coluna por algoritmo."""
    ns = sorted(res["N"].unique())
    algos = [a for a in ORDEM_ALGOS if a in set(res["algo"])]
    fig, eixos = plt.subplots(len(ns), len(algos), squeeze=False,
                              figsize=(5.4 * len(algos), 4.2 * len(ns)),
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


def _linha(ax, d, fw, y, metrica, desvio=None):
    """Uma curva de um framework. Marcador cheio p/ ms_total, vazado p/ ms_parede."""
    est = ESTILO_METRICA[metrica]
    cheio = metrica == "total"
    kwargs = dict(color=COR[fw], marker=MARCA[fw], markersize=7, ls=est["ls"], alpha=est["alpha"],
                  markerfacecolor=COR[fw] if cheio else "white",
                  label=f"{fw} ({ROTULO_METRICA[metrica]})")
    if desvio is not None and d[desvio].max() > 0:
        ax.errorbar(d["p"], d[y], yerr=d[desvio], capsize=3, **kwargs)
    else:
        ax.plot(d["p"], d[y], **kwargs)


def _desenhar_grade(res, titulo, ylabel, prefixo, ideal, usar_erro=False, log=False):
    fig, eixos, ns, algos = _grade(res, titulo)

    for i, n in enumerate(ns):
        for j, algo in enumerate(algos):
            ax = eixos[i][j]
            sub = res[(res["N"] == n) & (res["algo"] == algo)]
            procs = sorted(sub["p"].unique())

            if ideal:
                ideal(ax, procs)

            for fw in ["MPI", "Spark"]:
                d = sub[sub["framework"] == fw].sort_values("p")
                if d.empty:
                    continue
                for metrica in METRICAS:
                    y = f"{prefixo}_{metrica}" if prefixo else METRICAS[metrica]
                    if d[y].notna().any():
                        _linha(ax, d, fw, y, metrica,
                               desvio=f"desvio_{metrica}" if usar_erro else None)

            if log:
                ax.set_yscale("log")
            _acabamento(ax, n, algo, procs, ylabel, j == 0)
            if i == 0 and j == 0:
                ax.legend(fontsize=8)
    return fig


def grafico_tempo(res):
    return _desenhar_grade(res, "Tempo de execução: MPI vs Spark", "tempo (ms, escala log)",
                           prefixo=None, ideal=None, usar_erro=True, log=True)


def grafico_speedup(res):
    def ideal(ax, procs):
        ax.plot(procs, procs, color="gray", ls=":", label="ideal (linear)")
    return _desenhar_grade(res, "Speedup: MPI vs Spark", "speedup  T(1) / T(p)",
                           prefixo="speedup", ideal=ideal)


def grafico_eficiencia(res):
    def ideal(ax, _procs):
        ax.axhline(1.0, color="gray", ls=":", label="ideal (100%)")
    fig = _desenhar_grade(res, "Eficiência: MPI vs Spark", "eficiência  speedup / p",
                          prefixo="eficiencia", ideal=ideal)
    for ax in fig.axes:
        ax.set_ylim(bottom=0)
    return fig


def grafico_comparacao_direta(razao):
    """MPI/Spark no mesmo p: acima da linha 1, o Spark foi mais rápido."""
    fig, eixos, ns, algos = _grade(razao, "Comparação direta no mesmo p: tempo MPI / tempo Spark")

    for i, n in enumerate(ns):
        for j, algo in enumerate(algos):
            ax = eixos[i][j]
            d = razao[(razao["N"] == n) & (razao["algo"] == algo)].sort_values("p")
            procs = sorted(d["p"].unique())

            ax.axhline(1.0, color="gray", ls=":", label="empate")
            for metrica in METRICAS:
                y = f"razao_{metrica}"
                if d[y].notna().any():
                    est = ESTILO_METRICA[metrica]
                    ax.plot(d["p"], d[y], color="tab:green", marker="D", markersize=6,
                            ls=est["ls"], alpha=est["alpha"],
                            markerfacecolor="tab:green" if metrica == "total" else "white",
                            label=ROTULO_METRICA[metrica])

            _acabamento(ax, n, algo, procs, "MPI / Spark  (>1: Spark mais rápido)", j == 0)
            if i == 0 and j == 0:
                ax.legend(fontsize=8)
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

    res = calcular_metricas(agregar(carregar(args.mpi, args.spark), args.estatistica))
    razao = calcular_razao(res)

    os.makedirs(args.saida, exist_ok=True)
    res.to_csv(os.path.join(args.saida, "resumo.csv"), index=False)
    print(res.round(3).to_string(index=False))
    if not razao.empty:
        print("\nRazão MPI/Spark (>1: Spark mais rápido):")
        print(razao.round(3).to_string(index=False))

    figuras = {
        "tempo": grafico_tempo(res),
        "speedup": grafico_speedup(res),
        "eficiencia": grafico_eficiencia(res),
    }
    if not razao.empty:
        figuras["comparacao_direta"] = grafico_comparacao_direta(razao)

    for nome, fig in figuras.items():
        caminho = os.path.join(args.saida, f"{nome}.png")
        fig.savefig(caminho, dpi=150)
        print(f"salvo: {caminho}")

    if not args.sem_janela:
        plt.show()


if __name__ == "__main__":
    main()