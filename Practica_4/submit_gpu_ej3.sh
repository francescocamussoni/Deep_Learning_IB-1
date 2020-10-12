#! /bin/bash
#$ -N output_3
#$ -cwd
#$ -j y
#$ -S /bin/bash
#$ -V
##  pido la cola gpu.q
#$ -q gpu@compute-6-7.local
## pido una placa
#$ -l gpu=1
#$ -l memoria_a_usar=1G
#
# Load gpu drivers and conda
module load miniconda

source activate deep_learning

# Execute the script
hostname

#python testkeras.py
# python ej_03.py -lr 1e-5 -rf 1e-1 -e 200 -bs 500
# python ej_03.py -lr 1e-5 -rf 1e-2 -e 200 -bs 500
# python ej_03.py -lr 1e-5 -rf 1e-3 -e 200 -bs 500
# python ej_03.py -lr 1e-5 -rf 0 -e 200 -bs 500

# python ej_03.py -lr 1e-4 -rf 1e-1 -e 200 -bs 500
# python ej_03.py -lr 1e-4 -rf 1e-2 -e 200 -bs 500
# python ej_03.py -lr 1e-4 -rf 1e-3 -e 200 -bs 500
# python ej_03.py -lr 1e-4 -rf 0 -e 200 -bs 500

# python ej_03.py -lr 1e-3 -rf 1e-1 -e 200 -bs 500
# python ej_03.py -lr 1e-3 -rf 1e-2 -e 200 -bs 500
# python ej_03.py -lr 1e-3 -rf 1e-3 -e 200 -bs 500
# python ej_03.py -lr 1e-3 -rf 0 -e 200 -bs 500

# python ej_03.py -lr 1e-2 -rf 1e-1 -e 200 -bs 500
# python ej_03.py -lr 1e-2 -rf 1e-2 -e 200 -bs 500
# python ej_03.py -lr 1e-2 -rf 1e-3 -e 200 -bs 500
# python ej_03.py -lr 1e-2 -rf 0 -e 200 -bs 500

# Estas son para probar diferentes BN
# python ej_03.py -lr 1e-2 -rf 0 -e 200 -bs 16
# python ej_03.py -lr 1e-2 -rf 0 -e 200 -bs 32
# python ej_03.py -lr 1e-2 -rf 0 -e 200 -bs 64
# python ej_03.py -lr 1e-2 -rf 0 -e 200 -bs 128
# python ej_03.py -lr 1e-2 -rf 0 -e 200 -bs 256
# python ej_03.py -lr 1e-2 -rf 0 -e 200 -bs 512
# python ej_03.py -lr 1e-2 -rf 0 -e 200 -bs 1024

# Estos son para el ejercicio 3 con Dropout
python ej_03_Dropout.py -lr 1e-5 -rf 0 -e 200 -bs 500 -do 0
python ej_03_Dropout.py -lr 1e-5 -rf 0 -e 200 -bs 500 -do 0.25
python ej_03_Dropout.py -lr 1e-5 -rf 0 -e 200 -bs 500 -do 0.5
python ej_03_Dropout.py -lr 1e-5 -rf 0 -e 200 -bs 500 -do 0.75
python ej_03_Dropout.py -lr 1e-5 -rf 0 -e 200 -bs 500 -do 1

python ej_03_Dropout.py -lr 1e-4 -rf 0 -e 200 -bs 500 -do 0
python ej_03_Dropout.py -lr 1e-4 -rf 0 -e 200 -bs 500 -do 0.25
python ej_03_Dropout.py -lr 1e-4 -rf 0 -e 200 -bs 500 -do 0.5
python ej_03_Dropout.py -lr 1e-4 -rf 0 -e 200 -bs 500 -do 0.75
python ej_03_Dropout.py -lr 1e-4 -rf 0 -e 200 -bs 500 -do 1

python ej_03_Dropout.py -lr 1e-3 -rf 0 -e 200 -bs 500 -do 0
python ej_03_Dropout.py -lr 1e-3 -rf 0 -e 200 -bs 500 -do 0.25
python ej_03_Dropout.py -lr 1e-3 -rf 0 -e 200 -bs 500 -do 0.5
python ej_03_Dropout.py -lr 1e-3 -rf 0 -e 200 -bs 500 -do 0.75
python ej_03_Dropout.py -lr 1e-3 -rf 0 -e 200 -bs 500 -do 1

python ej_03_Dropout.py -lr 1e-2 -rf 0 -e 200 -bs 500 -do 0
python ej_03_Dropout.py -lr 1e-2 -rf 0 -e 200 -bs 500 -do 0.25
python ej_03_Dropout.py -lr 1e-2 -rf 0 -e 200 -bs 500 -do 0.5
python ej_03_Dropout.py -lr 1e-2 -rf 0 -e 200 -bs 500 -do 0.75
python ej_03_Dropout.py -lr 1e-2 -rf 0 -e 200 -bs 500 -do 1
