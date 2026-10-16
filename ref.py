from mpi4py import MPI
import numpy as np

def print_matrix(matrix):
    for row in matrix:
        print(" ".join(str(element) for element in row))

def print_vector(vector):
    print(" ".join(str(element) for element in vector))

comm = MPI.COMM_WORLD
num_procs = comm.Get_size()
proc_id = comm.Get_rank()

matrix_size = 0
matrix = None
vector = None
local_result = None
result = None

if proc_id == 0:
    # Solicita o tamanho da matriz ao usuário
    matrix_size = int(input("Informe o tamanho da matriz: "))

    # Verifica se é divisível pelo número de processos
    if matrix_size % num_procs != 0:
        print(f"Erro: O tamanho da matriz ({matrix_size}) deve ser divisível pelo número de processos ({num_procs}).")
        MPI.Finalize()
        exit()

    # Gera a matriz com valores aleatórios
    np.random.seed(0)
    matrix = np.random.randint(0, 10, (matrix_size, matrix_size)).astype(np.intc)

    # Inicializa o vetor com valores a partir de 1
    vector = np.arange(1, matrix_size + 1, dtype=np.intc)

# Envia o tamanho da matriz
matrix_size = comm.bcast(matrix_size, root=0)

# Define linhas por processo
rows_per_proc = matrix_size // num_procs

# Buffer para receber submatriz (1D)
recv_buffer = np.zeros(rows_per_proc * matrix_size, dtype=np.intc)

# Flatten apenas no processo 0
flat_matrix = matrix.flatten().astype(np.intc) if proc_id == 0 else None

# Scatter da matriz achatada
comm.Scatter(flat_matrix, recv_buffer, root=0)

# Reformata a submatriz
submatrix = recv_buffer.reshape((rows_per_proc, matrix_size))

# Aloca e transmite o vetor
if proc_id != 0:
    vector = np.zeros(matrix_size, dtype=np.intc)

comm.Bcast(vector, root=0)

# Multiplicação local
local_result = np.matmul(submatrix, vector)


# Processo 0 coleta resultados
if proc_id == 0:
    result = np.zeros(matrix_size, dtype=np.intc)

comm.Gather(local_result, result, root=0)

# Saída final
if proc_id == 0:
    print("\nMatriz:")
    print_matrix(matrix)
    print("\nVetor:")
    print_vector(vector)
    print("\nResultado:")
    print_vector(result)

MPI.Finalize()