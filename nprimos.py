# primos_mpi.py
from mpi4py import MPI
import numpy as np

# Trivial
def trivial_prime_numbers(list_n: list):
    list_of_primes = []
    for n in list_n:
        if(trivial_is_prime(n)):
            list_of_primes.append(n)

    return list_of_primes


def trivial_is_prime(n):
    if n<=1:
        return False
    
    for i in range(2, n):
        if n % i == 0:
            return False
    
    return True


# Trivial: menores que sqrt(n)

# Crivo de Eratóstenes

# Fazer com que o MPI rode esses algotimos

def main():
    # [1, 10] = range(1, 10+1)
    list_of_n = range(1, 11+1)
    list_of_prime_n = trivial_prime_numbers(list_of_n)

    print(list_of_prime_n)
    print("Number of prime numbers found: " + str(len(list_of_prime_n)))
    print()
main()

