#! /bin/bash
#$ -N o_6
#$ -cwd
#$ -j y
#$ -S /bin/bash
#$ -V
##  pido la cola gpu.q
#$ -q gpu
## pido una placa
#$ -l gpu=1
#$ -l memoria_a_usar=1G
#
# Load gpu drivers and conda
module load miniconda

source activate deep_learning

# Execute the script
hostname

# Ejercicio 6, con dos capas de 20 neuronas
python ej_06.py -lr 1e-5 -rf 1e-1 -e 500 -bs 32 -nn 20
python ej_06.py -lr 1e-5 -rf 1e-2 -e 500 -bs 32 -nn 20
python ej_06.py -lr 1e-5 -rf 1e-3 -e 500 -bs 32 -nn 20
python ej_06.py -lr 1e-5 -rf 0    -e 500 -bs 32 -nn 20

python ej_06.py -lr 1e-4 -rf 1e-1 -e 500 -bs 32 -nn 20
python ej_06.py -lr 1e-4 -rf 1e-2 -e 500 -bs 32 -nn 20
python ej_06.py -lr 1e-4 -rf 1e-3 -e 500 -bs 32 -nn 20
python ej_06.py -lr 1e-4 -rf 0    -e 500 -bs 32 -nn 20

python ej_06.py -lr 1e-3 -rf 1e-1 -e 500 -bs 32 -nn 20
python ej_06.py -lr 1e-3 -rf 1e-2 -e 500 -bs 32 -nn 20
python ej_06.py -lr 1e-3 -rf 1e-3 -e 500 -bs 32 -nn 20
python ej_06.py -lr 1e-3 -rf 0    -e 500 -bs 32 -nn 20

python ej_06.py -lr 1e-2 -rf 1e-1 -e 500 -bs 32 -nn 20
python ej_06.py -lr 1e-2 -rf 1e-2 -e 500 -bs 32 -nn 20
python ej_06.py -lr 1e-2 -rf 1e-3 -e 500 -bs 32 -nn 20
python ej_06.py -lr 1e-2 -rf 0    -e 500 -bs 32 -nn 20

python ej_06.py -lr 1e-1 -rf 1e-1 -e 500 -bs 32 -nn 20
python ej_06.py -lr 1e-1 -rf 1e-2 -e 500 -bs 32 -nn 20
python ej_06.py -lr 1e-1 -rf 1e-3 -e 500 -bs 32 -nn 20
python ej_06.py -lr 1e-1 -rf 0    -e 500 -bs 32 -nn 20

######################
# python ej_06.py -lr 1e-5 -rf 1e-1 -e 500 -bs 64 -nn 20
# python ej_06.py -lr 1e-5 -rf 1e-2 -e 500 -bs 64 -nn 20
# python ej_06.py -lr 1e-5 -rf 1e-3 -e 500 -bs 64 -nn 20
# python ej_06.py -lr 1e-5 -rf 0    -e 500 -bs 64 -nn 20

# python ej_06.py -lr 1e-4 -rf 1e-1 -e 500 -bs 64 -nn 20
# python ej_06.py -lr 1e-4 -rf 1e-2 -e 500 -bs 64 -nn 20
# python ej_06.py -lr 1e-4 -rf 1e-3 -e 500 -bs 64 -nn 20
# python ej_06.py -lr 1e-4 -rf 0    -e 500 -bs 64 -nn 20

# python ej_06.py -lr 1e-3 -rf 1e-1 -e 500 -bs 64 -nn 20
# python ej_06.py -lr 1e-3 -rf 1e-2 -e 500 -bs 64 -nn 20
# python ej_06.py -lr 1e-3 -rf 1e-3 -e 500 -bs 64 -nn 20
# python ej_06.py -lr 1e-3 -rf 0    -e 500 -bs 64 -nn 20

# python ej_06.py -lr 1e-2 -rf 1e-1 -e 500 -bs 64 -nn 20
# python ej_06.py -lr 1e-2 -rf 1e-2 -e 500 -bs 64 -nn 20
# python ej_06.py -lr 1e-2 -rf 1e-3 -e 500 -bs 64 -nn 20
# python ej_06.py -lr 1e-2 -rf 0    -e 500 -bs 64 -nn 20

# python ej_06.py -lr 1e-1 -rf 1e-1 -e 500 -bs 64 -nn 20
# python ej_06.py -lr 1e-1 -rf 1e-2 -e 500 -bs 64 -nn 20
# python ej_06.py -lr 1e-1 -rf 1e-3 -e 500 -bs 64 -nn 20
# python ej_06.py -lr 1e-1 -rf 0    -e 500 -bs 64 -nn 20


# Ejercicio 6, con dos capas de 40 neuronas
python ej_06.py -lr 1e-5 -rf 1e-1 -e 500 -bs 32 -nn 40
python ej_06.py -lr 1e-5 -rf 1e-2 -e 500 -bs 32 -nn 40
python ej_06.py -lr 1e-5 -rf 1e-3 -e 500 -bs 32 -nn 40
python ej_06.py -lr 1e-5 -rf 0    -e 500 -bs 32 -nn 40

python ej_06.py -lr 1e-4 -rf 1e-1 -e 500 -bs 32 -nn 40
python ej_06.py -lr 1e-4 -rf 1e-2 -e 500 -bs 32 -nn 40
python ej_06.py -lr 1e-4 -rf 1e-3 -e 500 -bs 32 -nn 40
python ej_06.py -lr 1e-4 -rf 0    -e 500 -bs 32 -nn 40

python ej_06.py -lr 1e-3 -rf 1e-1 -e 500 -bs 32 -nn 40
python ej_06.py -lr 1e-3 -rf 1e-2 -e 500 -bs 32 -nn 40
python ej_06.py -lr 1e-3 -rf 1e-3 -e 500 -bs 32 -nn 40
python ej_06.py -lr 1e-3 -rf 0    -e 500 -bs 32 -nn 40

python ej_06.py -lr 1e-2 -rf 1e-1 -e 500 -bs 32 -nn 40
python ej_06.py -lr 1e-2 -rf 1e-2 -e 500 -bs 32 -nn 40
python ej_06.py -lr 1e-2 -rf 1e-3 -e 500 -bs 32 -nn 40
python ej_06.py -lr 1e-2 -rf 0    -e 500 -bs 32 -nn 40

python ej_06.py -lr 1e-1 -rf 1e-1 -e 500 -bs 32 -nn 40
python ej_06.py -lr 1e-1 -rf 1e-2 -e 500 -bs 32 -nn 40
python ej_06.py -lr 1e-1 -rf 1e-3 -e 500 -bs 32 -nn 40
python ej_06.py -lr 1e-1 -rf 0    -e 500 -bs 32 -nn 40

######################
# python ej_06.py -lr 1e-5 -rf 1e-1 -e 500 -bs 64 -nn 40
# python ej_06.py -lr 1e-5 -rf 1e-2 -e 500 -bs 64 -nn 40
# python ej_06.py -lr 1e-5 -rf 1e-3 -e 500 -bs 64 -nn 40
# python ej_06.py -lr 1e-5 -rf 0    -e 500 -bs 64 -nn 40

# python ej_06.py -lr 1e-4 -rf 1e-1 -e 500 -bs 64 -nn 40
# python ej_06.py -lr 1e-4 -rf 1e-2 -e 500 -bs 64 -nn 40
# python ej_06.py -lr 1e-4 -rf 1e-3 -e 500 -bs 64 -nn 40
# python ej_06.py -lr 1e-4 -rf 0    -e 500 -bs 64 -nn 40

# python ej_06.py -lr 1e-3 -rf 1e-1 -e 500 -bs 64 -nn 40
# python ej_06.py -lr 1e-3 -rf 1e-2 -e 500 -bs 64 -nn 40
# python ej_06.py -lr 1e-3 -rf 1e-3 -e 500 -bs 64 -nn 40
# python ej_06.py -lr 1e-3 -rf 0    -e 500 -bs 64 -nn 40

# python ej_06.py -lr 1e-2 -rf 1e-1 -e 500 -bs 64 -nn 40
# python ej_06.py -lr 1e-2 -rf 1e-2 -e 500 -bs 64 -nn 40
# python ej_06.py -lr 1e-2 -rf 1e-3 -e 500 -bs 64 -nn 40
# python ej_06.py -lr 1e-2 -rf 0    -e 500 -bs 64 -nn 40

# python ej_06.py -lr 1e-1 -rf 1e-1 -e 500 -bs 64 -nn 40
# python ej_06.py -lr 1e-1 -rf 1e-2 -e 500 -bs 64 -nn 40
# python ej_06.py -lr 1e-1 -rf 1e-3 -e 500 -bs 64 -nn 40
# python ej_06.py -lr 1e-1 -rf 0    -e 500 -bs 64 -nn 40



