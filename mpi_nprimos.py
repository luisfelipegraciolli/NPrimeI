# primos_mpi.py
from mpi4py import MPI
import numpy as np
import sys
import math
import os
import csv

N_PADRAO = 1_000
CSV_ARQUIVO = "mpi.csv"


def prime_nums_interval(is_prime_function, inicio, fim):
    list_of_primes = []
    for n in range(inicio, fim):
        if(is_prime_function(n)):
            list_of_primes.append(n)

    return list_of_primes



# Trivial: menores que sqrt(n)

def is_prime_sqrt(n):
    if n <= 1:
        return False
    
    # Checa só até a raiz quadrada desse numero
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            return False
            
    return True

def primos_trivial(inicio, fim):
    return prime_nums_interval(is_prime_trivial, inicio, fim)
 
 
def primos_raiz(inicio, fim):
    return prime_nums_interval(is_prime_sqrt, inicio, fim)



# Crivo de Eratóstenes O(n)
def primos_ate(n):
    if n < 2:
        return []
    
    eh_primo = [True] * (n + 1) # Inicializa tudo como true
    eh_primo[0] = False
    eh_primo[1] = False

    limite = math.floor(math.sqrt(n))

    for i in range(2, limite):
        if eh_primo[i]:
            for j in range(i*i, n + 1, i):
                eh_primo[j] = False

    primos = []
    for n in range(2, n+1):
        if(eh_primo[n]):
            primos.append(n)
    
    return primos

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


# Fazer com que o MPI rode esses algotimos
ALGORITMOS = {
    "raiz": primos_raiz,
    "crivo": primos_crivo,
}


def consolidar(partes, tempos_ms):
    """Junta os resultados de todos os processos para UM algoritmo.
 
    partes: lista com os primos de cada processo (na ordem dos ranks)
    tempos_ms: lista com o tempo de cada processo, em ms
 
    Como os segmentos são contíguos e crescentes, basta concatenar as partes.
    O tempo "oficial" é o do processo mais lento, pois é ele que faz o
    programa terminar.
    """
    primos = [p for parte in partes for p in parte]
    return {
        "primos": primos,
        "total_ms": max(tempos_ms),
        "tempos_ms": tempos_ms,
    }

def todos_iguais(consolidados):
    """True se todos os algoritmos encontraram exatamente os mesmos primos."""
    listas = [c["primos"] for c in consolidados.values()]
    return all(lista == listas[0] for lista in listas)



def imprimir_resumo(consolidados, n, quantidade_processos):
    print(f"\nIntervalo: 1..{n} | Processos: {quantidade_processos}\n")
 
    print(f"{'algoritmo':<10} {'total (ms)':>12} {'primos':>8}   tempo por processo (ms)")
    print("-" * 72)
    for nome, c in consolidados.items():
        por_processo = "  ".join(f"{t:.1f}" for t in c["tempos_ms"])
        print(f"{nome:<10} {c['total_ms']:>12.1f} {len(c['primos']):>8}   {por_processo}")
 
    print()
    if todos_iguais(consolidados):
        print("Os três algoritmos encontraram os mesmos primos.")
    else:
        print("ATENÇÃO: os algoritmos encontraram primos diferentes!")

def salvar_csv(consolidados, n, size, caminho=CSV_ARQUIVO):
    """Acrescenta uma linha por algoritmo em 'caminho': algo, ms_total, n.
 
    O cabeçalho só é escrito se o arquivo ainda não existe (ou está vazio),
    então várias execuções com N diferentes ficam no mesmo arquivo.
    """
    arquivo_novo = not os.path.exists(caminho) or os.path.getsize(caminho) == 0
 
    with open(caminho, "a", newline="") as arquivo:
        escritor = csv.writer(arquivo)
        if arquivo_novo:
            escritor.writerow(["framework", "algo", "ms_total", "N", "N_processos"])
        for nome, c in consolidados.items():
            escritor.writerow(["MPI", nome, round(c["total_ms"], 3), n, size])


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
        t0 = MPI.Wtime()
        primos_locais = algoritmo(inicio, fim)
        ms = (MPI.Wtime() - t0) * 1000

        tempos_ms = comm.gather(ms, root=0)
        partes = comm.gather(primos_locais, root=0)

        if rank == 0:
            consolidados[nome] = consolidar(partes, tempos_ms)

    
    if rank == 0:
        salvar_csv(consolidados, n, size)
        imprimir_resumo(consolidados, n, size)


    # Junte partes numa lista única e guarde por algoritmo.
    # O tempo "oficial" de um algoritmo é o do processo mais lento: max(tempos). Guarde também a lista tempos para ver cada processo.
    # No final, confira que os três algoritmos produziram exatamente o mesmo resultado, com um assert ou uma comparação simples.

    


if __name__ == "__main__":
    main()
