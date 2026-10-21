# primos_mpi.py
#
# Uso:
#   mpirun -n 4 python primos_mpi.py 1000000
#
# Métricas registradas por algoritmo (em ms):
#   ms_total  : tempo do processo MAIS LENTO só calculando os primos (entre o Barrier e o fim do algoritmo).
#               Equivale a "ms_total" do Spark (tarefa mais lenta).
#   ms_parede : tempo desde a largada (após o Barrier) até o rank 0 ter recebido TODOS os primos
#               (inclui os dois gather). Equivale a "ms_parede" do Spark (tempo visto pelo driver).

from mpi4py import MPI
import numpy as np
import sys
import math
import os
import csv

N_PADRAO = 1_000
CSV_ARQUIVO = "mpi.csv"
CABECALHO_CSV = ["framework", "algo", "ms_total", "ms_parede", "N", "N_processos"]


def prime_nums_interval(is_prime_function, inicio, fim):
    return [n for n in range(inicio, fim) if is_prime_function(n)]


def is_prime_sqrt(n):
    if n <= 1:
        return False

    # Checa só até a raiz quadrada desse número
    for i in range(2, math.isqrt(n) + 1):
        if n % i == 0:
            return False

    return True


def primos_trivial(inicio, fim):
    return prime_nums_interval(is_prime_trivial, inicio, fim)


def primos_raiz(inicio, fim):
    return prime_nums_interval(is_prime_sqrt, inicio, fim)


# Crivo de Eratóstenes clássico: O(n log log n)
def primos_ate(n):
    if n < 2:
        return []

    eh_primo = [True] * (n + 1)  # Inicializa tudo como True
    eh_primo[0] = False
    eh_primo[1] = False

    # +1 para incluir a própria raiz (antes o range parava um número antes)
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

    # O maior número do segmento é fim-1, então todo composto dele tem
    # um divisor primo <= raiz de (fim-1).
    primos_base = primos_ate(math.isqrt(fim - 1))

    # eh_primo[i] refere-se ao número (inicio + i)
    eh_primo = [True] * (fim - inicio)

    for p in primos_base:
        # Primeiro múltiplo de p que está dentro do segmento
        primeiro_multiplo = ((inicio + p - 1) // p) * p
        # Nunca riscar o próprio p, então começamos em no mínimo p*p
        primeiro = max(p * p, primeiro_multiplo)

        for multiplo in range(primeiro, fim, p):
            eh_primo[multiplo - inicio] = False

    return [inicio + i for i, ok in enumerate(eh_primo) if ok]


# Algoritmos que o MPI vai rodar
ALGORITMOS = {
    "raiz": primos_raiz,
    "crivo": primos_crivo,
}


def consolidar(partes, tempos_ms, ms_parede):
    """Junta os resultados de todos os processos para UM algoritmo.

    partes: lista com os primos de cada processo (na ordem dos ranks)
    tempos_ms: lista com o tempo de computação de cada processo, em ms
    ms_parede: tempo ponta a ponta medido no rank 0 (inclui os gather), em ms

    Como os segmentos são contíguos e crescentes, basta concatenar as partes.
    O tempo "oficial" (total_ms) é o do processo mais lento, pois é ele que
    faz o programa terminar.
    """
    primos = [p for parte in partes for p in parte]
    return {
        "primos": primos,
        "total_ms": max(tempos_ms),
        "parede_ms": ms_parede,
        "tempos_ms": tempos_ms,
    }


def todos_iguais(consolidados):
    """True se todos os algoritmos encontraram exatamente os mesmos primos."""
    listas = [c["primos"] for c in consolidados.values()]
    return all(lista == listas[0] for lista in listas)


def imprimir_resumo(consolidados, n, quantidade_processos):
    print(f"\nIntervalo: 1..{n} | Processos: {quantidade_processos}\n")

    print(f"{'algoritmo':<10} {'total (ms)':>12} {'parede (ms)':>12} {'primos':>8}   tempo por processo (ms)")
    print("-" * 85)
    for nome, c in consolidados.items():
        por_processo = "  ".join(f"{t:.1f}" for t in c["tempos_ms"])
        print(f"{nome:<10} {c['total_ms']:>12.1f} {c['parede_ms']:>12.1f} {len(c['primos']):>8}   {por_processo}")

    print()
    if todos_iguais(consolidados):
        print("Os três algoritmos encontraram os mesmos primos.")
    else:
        print("ATENÇÃO: os algoritmos encontraram primos diferentes!")


def salvar_csv(consolidados, n, size, caminho=CSV_ARQUIVO):
    """Acrescenta uma linha por algoritmo em 'caminho'.

    O cabeçalho só é escrito se o arquivo ainda não existe (ou está vazio),
    então várias execuções com N e processos diferentes ficam no mesmo arquivo.
    Se o arquivo existir com outro cabeçalho (ex.: versão antiga sem ms_parede),
    o programa para em vez de misturar formatos.
    """
    arquivo_novo = not os.path.exists(caminho) or os.path.getsize(caminho) == 0

    if not arquivo_novo:
        with open(caminho, newline="") as arquivo:
            cabecalho_existente = next(csv.reader(arquivo), [])
        if cabecalho_existente != CABECALHO_CSV:
            raise SystemExit(
                f"'{caminho}' tem outro formato de colunas ({cabecalho_existente}). "
                f"Renomeie ou apague o arquivo antigo para gravar o novo formato {CABECALHO_CSV}."
            )

    with open(caminho, "a", newline="") as arquivo:
        escritor = csv.writer(arquivo)
        if arquivo_novo:
            escritor.writerow(CABECALHO_CSV)
        for nome, c in consolidados.items():
            escritor.writerow(["MPI", nome, round(c["total_ms"], 3), round(c["parede_ms"], 3), n, size])


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # o rank 0 divide 1..N em segmentos iguais e distribui
    if rank == 0:
        n = int(sys.argv[1]) if len(sys.argv) > 1 else N_PADRAO
        segmentos = np.array_split(np.arange(1, n + 1), size)
    else:
        n = None
        segmentos = None

    n = comm.bcast(n, root=0)
    meu_segmento = comm.scatter(segmentos, root=0)

    if len(meu_segmento) == 0:  # acontece se N < número de processos
        inicio, fim = 1, 1
    else:
        inicio, fim = int(meu_segmento[0]), int(meu_segmento[-1]) + 1

    # roda cada algoritmo no mesmo segmento e mede o tempo
    consolidados = {}

    for nome, algoritmo in ALGORITMOS.items():
        comm.Barrier()
        t_inicio = MPI.Wtime()

        primos_locais = algoritmo(inicio, fim)
        ms = (MPI.Wtime() - t_inicio) * 1000  # só computação deste processo

        tempos_ms = comm.gather(ms, root=0)
        partes = comm.gather(primos_locais, root=0)

        # Largada (pós-Barrier) até o rank 0 ter todos os primos. Só é válido no rank 0.
        ms_parede = (MPI.Wtime() - t_inicio) * 1000

        if rank == 0:
            consolidados[nome] = consolidar(partes, tempos_ms, ms_parede)

    if rank == 0:
        salvar_csv(consolidados, n, size)
        imprimir_resumo(consolidados, n, size)


if __name__ == "__main__":
    main()