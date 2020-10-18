#! /bin/bash
#$ -N o_3_Regu
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

#python testkeras.py
python ej_03_Regularizers.py -lr 1e-5 -rf 1e-1 -e 200 -bs 256 -nn 25
python ej_03_Regularizers.py -lr 1e-5 -rf 1e-2 -e 200 -bs 256 -nn 25
python ej_03_Regularizers.py -lr 1e-5 -rf 1e-3 -e 200 -bs 256 -nn 25
python ej_03_Regularizers.py -lr 1e-5 -rf 0    -e 200 -bs 256 -nn 25

python ej_03_Regularizers.py -lr 1e-4 -rf 1e-1 -e 200 -bs 256 -nn 25
python ej_03_Regularizers.py -lr 1e-4 -rf 1e-2 -e 200 -bs 256 -nn 25
python ej_03_Regularizers.py -lr 1e-4 -rf 1e-3 -e 200 -bs 256 -nn 25
python ej_03_Regularizers.py -lr 1e-4 -rf 0    -e 200 -bs 256 -nn 25

python ej_03_Regularizers.py -lr 1e-3 -rf 1e-1 -e 200 -bs 256 -nn 25
python ej_03_Regularizers.py -lr 1e-3 -rf 1e-2 -e 200 -bs 256 -nn 25
python ej_03_Regularizers.py -lr 1e-3 -rf 1e-3 -e 200 -bs 256 -nn 25
python ej_03_Regularizers.py -lr 1e-3 -rf 0    -e 200 -bs 256 -nn 25

python ej_03_Regularizers.py -lr 1e-2 -rf 1e-1 -e 200 -bs 256 -nn 25
python ej_03_Regularizers.py -lr 1e-2 -rf 1e-2 -e 200 -bs 256 -nn 25
python ej_03_Regularizers.py -lr 1e-2 -rf 1e-3 -e 200 -bs 256 -nn 25
python ej_03_Regularizers.py -lr 1e-2 -rf 0    -e 200 -bs 256 -nn 25

# Ahora con 50 neuronas
python ej_03_Regularizers.py -lr 1e-5 -rf 1e-1 -e 200 -bs 256 -nn 50
python ej_03_Regularizers.py -lr 1e-5 -rf 1e-2 -e 200 -bs 256 -nn 50
python ej_03_Regularizers.py -lr 1e-5 -rf 1e-3 -e 200 -bs 256 -nn 50
python ej_03_Regularizers.py -lr 1e-5 -rf 0    -e 200 -bs 256 -nn 50

python ej_03_Regularizers.py -lr 1e-4 -rf 1e-1 -e 200 -bs 256 -nn 50
python ej_03_Regularizers.py -lr 1e-4 -rf 1e-2 -e 200 -bs 256 -nn 50
python ej_03_Regularizers.py -lr 1e-4 -rf 1e-3 -e 200 -bs 256 -nn 50
python ej_03_Regularizers.py -lr 1e-4 -rf 0    -e 200 -bs 256 -nn 50

python ej_03_Regularizers.py -lr 1e-3 -rf 1e-1 -e 200 -bs 256 -nn 50
python ej_03_Regularizers.py -lr 1e-3 -rf 1e-2 -e 200 -bs 256 -nn 50
python ej_03_Regularizers.py -lr 1e-3 -rf 1e-3 -e 200 -bs 256 -nn 50
python ej_03_Regularizers.py -lr 1e-3 -rf 0    -e 200 -bs 256 -nn 50

python ej_03_Regularizers.py -lr 1e-2 -rf 1e-1 -e 200 -bs 256 -nn 50
python ej_03_Regularizers.py -lr 1e-2 -rf 1e-2 -e 200 -bs 256 -nn 50
python ej_03_Regularizers.py -lr 1e-2 -rf 1e-3 -e 200 -bs 256 -nn 50
python ej_03_Regularizers.py -lr 1e-2 -rf 0    -e 200 -bs 256 -nn 50