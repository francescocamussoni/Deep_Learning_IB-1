#! /bin/bash
#$ -N o_100_VGG
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
# python ej_10_VGG16.py -lr 1e-2 -rf 1e-5 -e 100 -bs 64 --dataset 'cifar100'
# python ej_10_VGG16.py -lr 1e-2 -rf 1e-4 -e 100 -bs 64 --dataset 'cifar100'
# python ej_10_VGG16.py -lr 1e-2 -rf 1e-3 -e 100 -bs 64 --dataset 'cifar100'

python ej_10_VGG16.py -lr 1e-3 -rf 1e-5 -e 100 -bs 64 --dataset 'cifar100'
python ej_10_VGG16.py -lr 1e-3 -rf 1e-4 -e 100 -bs 64 --dataset 'cifar100'
python ej_10_VGG16.py -lr 1e-3 -rf 1e-3 -e 100 -bs 64 --dataset 'cifar100'

python ej_10_VGG16.py -lr 1e-4 -rf 1e-5 -e 100 -bs 64 --dataset 'cifar100'
python ej_10_VGG16.py -lr 1e-4 -rf 1e-4 -e 100 -bs 64 --dataset 'cifar100'
python ej_10_VGG16.py -lr 1e-4 -rf 1e-3 -e 100 -bs 64 --dataset 'cifar100'