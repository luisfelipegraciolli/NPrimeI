# primos_spark.py
#
# Uso:
#   python primos_spark.py 1000000 4          # N=1.000.000, 4 partições (local[*])
#   spark-submit primos_spark.py 1000000 4    # idem, deixando o spark-submit escolher o master
#
# Mapeamento MPI -> Spark:
#   processo (rank)        -> partição / tarefa (task)
#   bcast + scatter        -> sc.parallelize(segmentos, n_particoes)
#   gather                 -> rdd.collect()
#   Barrier + Wtime        -> medição dentro da tarefa (perf_counter) + tempo de parede no driver

import csv
import math
import os
import sys
import time

from pyspark.sql import SparkSession

N_PADRAO = 1_000
PARTICOES_PADRAO = 4
CSV_ARQUIVO = "spark.csv"


# ----------------------------------------------------------------------
# Algoritmos de primos (iguais aos da versão MPI)
# ----------------------------------------------------------------------

def prime_nums_interval(is_prime_function, inicio, fim):
    return [n for n in range(inicio, fim) if is_prime_function(n)]


def is_prime_sqrt(n):
    if n <= 1:
        return False
    for i in range(2, math.isqrt(n) + 1):
        if n % i == 0:
            return False
    return True


def primos_trivial(inicio, fim):
    return prime_nums_interval(is_prime_trivial, inicio, fim)


def primos_raiz(inicio, fim):
    return prime_nums_interval(is_prime_sqrt, inicio, fim)


def primos_ate(n):
    """Crivo de Eratóstenes clássico: primos em [2, n]."""
    if n < 2:
        return []

    eh_primo = [True] * (n + 1)
    eh_primo[0] = eh_primo[1] = False

    # +1 para incluir a própria raiz (na versão original o range parava antes dela)
    for i in range(2, math.isqrt(n) + 1):
        if eh_primo[i]:
            for j in range(i * i, n + 1, i):
                eh_primo[j] = False

    return [i for i in range(2, n + 1) if eh_primo[i]]


def primos_crivo(inicio, fim):
    """Crivo segmentado: primos em [inicio, fim)."""
    inicio = max(inicio, 2)
    if fim <= inicio:
        return []

    primos_base = primos_ate(math.isqrt(fim - 1))
    eh_primo = [True] * (fim - inicio)

    for p in primos_base:
        primeiro_multiplo = ((inicio + p - 1) // p) * p
        primeiro = max(p * p, primeiro_multiplo)
        for multiplo in range(primeiro, fim, p):
            eh_primo[multiplo - inicio] = False

    return [inicio + i for i, ok in enumerate(eh_primo) if ok]


ALGORITMOS = {
    "raiz": primos_raiz,
    "crivo": primos_crivo,
}


# ----------------------------------------------------------------------
# Divisão do intervalo e tarefa executada em cada partição
# ----------------------------------------------------------------------

def dividir_intervalo(n, particoes):
    """Divide 1..n em 'particoes' segmentos contíguos [inicio, fim),
    com tamanhos o mais iguais possível (mesmo comportamento do np.array_split)."""
    base, resto = divmod(n, particoes)
    segmentos = []
    inicio = 1
    for i in range(particoes):
        tamanho = base + (1 if i < resto else 0)
        segmentos.append((i, inicio, inicio + tamanho))
        inicio += tamanho
    return segmentos


def processar_segmento(args):
    """Roda em um executor: calcula os primos do segmento e mede o próprio tempo."""
    idx, inicio, fim, nome = args
    algoritmo = ALGORITMOS[nome]

    t0 = time.perf_counter()
    primos = algoritmo(inicio, fim) if fim > inicio else []
    ms = (time.perf_counter() - t0) * 1000

    return idx, primos, ms



def consolidar(resultados, ms_parede):
    """resultados: lista de (idx, primos, ms) de todas as partições.

    Os segmentos são contíguos e crescentes, então basta ordenar por idx e concatenar.
    O tempo "oficial" é o da tarefa mais lenta; ms_parede é o tempo total visto
    pelo driver (inclui overhead de escalonamento, serialização e collect).
    """
    resultados = sorted(resultados, key=lambda r: r[0])
    primos = [p for _, parte, _ in resultados for p in parte]
    tempos_ms = [ms for _, _, ms in resultados]
    return {
        "primos": primos,
        "total_ms": max(tempos_ms),
        "parede_ms": ms_parede,
        "tempos_ms": tempos_ms,
    }


def todos_iguais(consolidados):
    listas = [c["primos"] for c in consolidados.values()]
    return all(lista == listas[0] for lista in listas)


def imprimir_resumo(consolidados, n, particoes):
    print(f"\nIntervalo: 1..{n} | Partições: {particoes}\n")

    print(f"{'algoritmo':<10} {'tarefa mais lenta (ms)':>23} {'parede (ms)':>12} {'primos':>8}   tempo por partição (ms)")
    print("-" * 95)
    for nome, c in consolidados.items():
        por_particao = "  ".join(f"{t:.1f}" for t in c["tempos_ms"])
        print(f"{nome:<10} {c['total_ms']:>23.1f} {c['parede_ms']:>12.1f} {len(c['primos']):>8}   {por_particao}")

    print()
    if todos_iguais(consolidados):
        print("Os três algoritmos encontraram os mesmos primos.")
    else:
        print("ATENÇÃO: os algoritmos encontraram primos diferentes!")


def salvar_csv(consolidados, n, particoes, caminho=CSV_ARQUIVO):
    """Acrescenta uma linha por algoritmo em 'caminho'."""
    arquivo_novo = not os.path.exists(caminho) or os.path.getsize(caminho) == 0

    with open(caminho, "a", newline="") as arquivo:
        escritor = csv.writer(arquivo)
        if arquivo_novo:
            escritor.writerow(["framework", "algo", "ms_total", "ms_parede", "N", "N_particoes"])
        for nome, c in consolidados.items():
            escritor.writerow(["Spark", nome, round(c["total_ms"], 3), round(c["parede_ms"], 3), n, particoes])


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else N_PADRAO
    particoes = int(sys.argv[2]) if len(sys.argv) > 2 else PARTICOES_PADRAO

    spark = SparkSession.builder.appName("primos_spark").getOrCreate()
    sc = spark.sparkContext
    sc.setLogLevel("WARN")

    segmentos = dividir_intervalo(n, particoes)

    # Aquecimento: o primeiro job paga o custo de subir os workers Python,
    # o que distorceria a medição do primeiro algoritmo.
    sc.parallelize(range(particoes), particoes).map(lambda x: x).collect()

    consolidados = {}

    for nome in ALGORITMOS:
        tarefas = [(idx, ini, fim, nome) for idx, ini, fim in segmentos]

        # 1 elemento por partição => 1 tarefa por segmento
        rdd = sc.parallelize(tarefas, particoes).map(processar_segmento)

        t0 = time.perf_counter()
        resultados = rdd.collect()
        ms_parede = (time.perf_counter() - t0) * 1000

        consolidados[nome] = consolidar(resultados, ms_parede)

    salvar_csv(consolidados, n, particoes)
    imprimir_resumo(consolidados, n, particoes)

    spark.stop()


if __name__ == "__main__":
    main()
