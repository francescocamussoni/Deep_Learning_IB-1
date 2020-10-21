#! /bin/bash
#$ -N o_7
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
python ej_07_Falopeada_MSE.py -lr 1e-5 -rf 0 -e 1000 -bs 512
python ej_07_Falopeada_MSE.py -lr 1e-4 -rf 0 -e 1000 -bs 512
python ej_07_Falopeada_MSE.py -lr 1e-3 -rf 0 -e 1000 -bs 512




