# primos_mpi.py
from mpi4py import MPI
import numpy as np
import math

# Checa numeros primos em um intervalo com uma função desejada



def prime_nums_interval(is_prime_function, inicio, fim):
    list_of_primes = []
    for n in range(inicio, fim):
        if(is_prime_function(n)):
            list_of_primes.append(n)

    return list_of_primes


def is_prime_trivial(n):
    if n<=1:
        return False
    
    for i in range(2, n):
        if n % i == 0:
            return False
    
    return True


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

def main():
    # [inicio, fim)
    inicio = 1
    fim = 101

    list_of_prime_n0 = primos_trivial(inicio, fim)
    list_of_prime_n1 = primos_raiz(inicio, fim)
    list_of_prime_n2 = primos_crivo(inicio, fim)
    
    
    print(list_of_prime_n0)
    print(list_of_prime_n1)
    print(list_of_prime_n2)
    
main()

