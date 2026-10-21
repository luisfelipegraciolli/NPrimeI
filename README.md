# Números primos com MPI e Spark

Projeto para calcular números primos no intervalo de 1 a `N` usando processamento distribuído com MPI e Apache Spark, comparar o desempenho das duas abordagens e gerar gráficos a partir das medições.

Os dois programas principais executam os mesmos algoritmos em segmentos do intervalo:

- `raiz`: testa cada número por divisores até sua raiz quadrada.
- `crivo`: usa o crivo segmentado de Eratóstenes.

Ao final, cada programa informa os tempos medidos e a quantidade de primos encontrados. Os tempos e os parâmetros da execução também são acrescentados aos arquivos CSV para permitir várias medições.

## Arquivos

- `mpi_nprimos.py`: divide o intervalo entre processos MPI.
- `spark_nprimos.py`: divide o intervalo entre partições/tarefas Spark.
- `comparar_mpi_spark.py`: lê os CSVs, calcula métricas comparativas e cria gráficos.
- `mpi.csv` e `spark.csv`: dados de execução usados pelo comparador.
- `graficos/`: destino padrão dos gráficos e do resumo calculado.
- `ref.py`: exemplo separado de multiplicação distribuída de matriz por vetor com MPI; solicita o tamanho da matriz interativamente.

## Pré-requisitos

É necessário ter Python 3 instalado. MPI também exige uma implementação MPI disponível no sistema (por exemplo, Open MPI ou MPICH) e o pacote `mpi4py`. Para Spark, instale o PySpark e tenha Java configurado conforme os requisitos da versão utilizada. O comparador precisa de `pandas` e `matplotlib`.

Instale as dependências Python com:

```bash
python -m pip install numpy mpi4py pyspark pandas matplotlib
```

Os comandos abaixo devem ser executados a partir da pasta do projeto. Se estiver usando um ambiente virtual, ative-o antes.

## Executar com MPI

Informe o número de processos e, opcionalmente, o limite `N`:

```bash
mpirun -n 4 python mpi_nprimos.py 1000000
```

Esse exemplo procura primos no intervalo `1..1.000.000` usando quatro processos. Se `N` for omitido, o padrão é `1000`:

```bash
mpirun -n 4 python mpi_nprimos.py
```

Cada execução acrescenta uma linha por algoritmo a `mpi.csv`. O arquivo registra `framework`, `algo`, `ms_total`, `ms_parede`, `N` e `N_processos`. Se `mpi.csv` já existir com um cabeçalho incompatível, o programa interrompe a gravação para evitar misturar formatos.

## Executar com Spark

Passe o limite `N` e o número de partições. Ambos são opcionais e, por padrão, usam `N=1000` e quatro partições:

```bash
python spark_nprimos.py 1000000 4
```

Também é possível iniciar pelo `spark-submit`:

```bash
spark-submit spark_nprimos.py 1000000 4
```

O Spark executa localmente quando não há outro master configurado. Cada execução acrescenta uma linha por algoritmo a `spark.csv`, com as colunas `framework`, `algo`, `ms_total`, `ms_parede`, `N` e `N_particoes`.

## Comparar resultados

Depois de executar os dois programas, gere os gráficos e o resumo:

```bash
python comparar_mpi_spark.py
```

Por padrão, o comparador lê `mpi.csv` e `spark.csv`, grava os resultados em `graficos/` e abre as janelas dos gráficos. Para executar sem abrir janelas:

```bash
python comparar_mpi_spark.py --sem-janela
```

É possível indicar outros arquivos CSV e outra pasta de saída:

```bash
python comparar_mpi_spark.py \
    --mpi dados_mpi.csv \
    --spark dados_spark.csv \
    --saida resultados \
    --sem-janela
```

O programa gera `tempo.png`, `speedup.png`, `eficiencia.png`, `comparacao_direta.png` (quando há configurações comparáveis) e `resumo.csv`. Execuções repetidas da mesma configuração são agregadas pela média por padrão; altere a estatística com `--estatistica median` ou `--estatistica min`.

`ms_total` representa o tempo de computação do processo/tarefa mais lento. `ms_parede` representa o tempo de ponta a ponta: no MPI, até o processo principal receber os resultados; no Spark, o tempo do job até o `collect()` terminar. Speedup e eficiência são calculados em relação à execução com um processo/partição; se ela não estiver disponível, usa-se a menor configuração encontrada e é exibido um aviso.

## Exemplo MPI adicional

O script `ref.py` não participa da comparação de primos. Ele distribui a multiplicação de uma matriz quadrada por um vetor e pede o tamanho da matriz durante a execução:

```bash
mpirun -n 4 python ref.py
```

O tamanho informado precisa ser divisível pelo número de processos.