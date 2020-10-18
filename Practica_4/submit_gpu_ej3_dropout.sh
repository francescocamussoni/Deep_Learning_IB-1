#! /bin/bash
#$ -N o_3_Drop
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

# Estos son para el ejercicio 3 con Dropout
python ej_03_Dropout.py -lr 1e-5 -rf 0 -e 200 -bs 256 -nn 25 -do 0
python ej_03_Dropout.py -lr 1e-5 -rf 0 -e 200 -bs 256 -nn 25 -do 0.25
python ej_03_Dropout.py -lr 1e-5 -rf 0 -e 200 -bs 256 -nn 25 -do 0.5
python ej_03_Dropout.py -lr 1e-5 -rf 0 -e 200 -bs 256 -nn 25 -do 0.75

python ej_03_Dropout.py -lr 1e-4 -rf 0 -e 200 -bs 256 -nn 25 -do 0
python ej_03_Dropout.py -lr 1e-4 -rf 0 -e 200 -bs 256 -nn 25 -do 0.25
python ej_03_Dropout.py -lr 1e-4 -rf 0 -e 200 -bs 256 -nn 25 -do 0.5
python ej_03_Dropout.py -lr 1e-4 -rf 0 -e 200 -bs 256 -nn 25 -do 0.75

python ej_03_Dropout.py -lr 1e-3 -rf 0 -e 200 -bs 256 -nn 25 -do 0
python ej_03_Dropout.py -lr 1e-3 -rf 0 -e 200 -bs 256 -nn 25 -do 0.25
python ej_03_Dropout.py -lr 1e-3 -rf 0 -e 200 -bs 256 -nn 25 -do 0.5
python ej_03_Dropout.py -lr 1e-3 -rf 0 -e 200 -bs 256 -nn 25 -do 0.75

python ej_03_Dropout.py -lr 1e-2 -rf 0 -e 200 -bs 256 -nn 25 -do 0
python ej_03_Dropout.py -lr 1e-2 -rf 0 -e 200 -bs 256 -nn 25 -do 0.25
python ej_03_Dropout.py -lr 1e-2 -rf 0 -e 200 -bs 256 -nn 25 -do 0.5
python ej_03_Dropout.py -lr 1e-2 -rf 0 -e 200 -bs 256 -nn 25 -do 0.75

# Ahora con 50 neuronas
python ej_03_Dropout.py -lr 1e-5 -rf 0 -e 200 -bs 256 -nn 50 -do 0
python ej_03_Dropout.py -lr 1e-5 -rf 0 -e 200 -bs 256 -nn 50 -do 0.25
python ej_03_Dropout.py -lr 1e-5 -rf 0 -e 200 -bs 256 -nn 50 -do 0.5
python ej_03_Dropout.py -lr 1e-5 -rf 0 -e 200 -bs 256 -nn 50 -do 0.75

python ej_03_Dropout.py -lr 1e-4 -rf 0 -e 200 -bs 256 -nn 50 -do 0
python ej_03_Dropout.py -lr 1e-4 -rf 0 -e 200 -bs 256 -nn 50 -do 0.25
python ej_03_Dropout.py -lr 1e-4 -rf 0 -e 200 -bs 256 -nn 50 -do 0.5
python ej_03_Dropout.py -lr 1e-4 -rf 0 -e 200 -bs 256 -nn 50 -do 0.75

python ej_03_Dropout.py -lr 1e-3 -rf 0 -e 200 -bs 256 -nn 50 -do 0
python ej_03_Dropout.py -lr 1e-3 -rf 0 -e 200 -bs 256 -nn 50 -do 0.25
python ej_03_Dropout.py -lr 1e-3 -rf 0 -e 200 -bs 256 -nn 50 -do 0.5
python ej_03_Dropout.py -lr 1e-3 -rf 0 -e 200 -bs 256 -nn 50 -do 0.75

python ej_03_Dropout.py -lr 1e-2 -rf 0 -e 200 -bs 256 -nn 50 -do 0
python ej_03_Dropout.py -lr 1e-2 -rf 0 -e 200 -bs 256 -nn 50 -do 0.25
python ej_03_Dropout.py -lr 1e-2 -rf 0 -e 200 -bs 256 -nn 50 -do 0.5
python ej_03_Dropout.py -lr 1e-2 -rf 0 -e 200 -bs 256 -nn 50 -do 0.75
