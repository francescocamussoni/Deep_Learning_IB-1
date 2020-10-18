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
python ej_03_BN.py -lr 1e-5 -e 200 -bs 64  -nn 25
python ej_03_BN.py -lr 1e-5 -e 200 -bs 128 -nn 25
python ej_03_BN.py -lr 1e-5 -e 200 -bs 256 -nn 25
python ej_03_BN.py -lr 1e-5 -e 200 -bs 512 -nn 25
python ej_03_BN.py -lr 1e-5 -e 200 -bs 1024 -nn 25

python ej_03_BN.py -lr 1e-4 -e 200 -bs 64  -nn 25
python ej_03_BN.py -lr 1e-4 -e 200 -bs 128 -nn 25
python ej_03_BN.py -lr 1e-4 -e 200 -bs 256 -nn 25
python ej_03_BN.py -lr 1e-4 -e 200 -bs 512 -nn 25
python ej_03_BN.py -lr 1e-4 -e 200 -bs 1024 -nn 25

python ej_03_BN.py -lr 1e-3 -e 200 -bs 64  -nn 25
python ej_03_BN.py -lr 1e-3 -e 200 -bs 128 -nn 25
python ej_03_BN.py -lr 1e-3 -e 200 -bs 256 -nn 25
python ej_03_BN.py -lr 1e-3 -e 200 -bs 512 -nn 25
python ej_03_BN.py -lr 1e-3 -e 200 -bs 1024 -nn 25

python ej_03_BN.py -lr 1e-2 -e 200 -bs 64  -nn 25
python ej_03_BN.py -lr 1e-2 -e 200 -bs 128 -nn 25
python ej_03_BN.py -lr 1e-2 -e 200 -bs 256 -nn 25
python ej_03_BN.py -lr 1e-2 -e 200 -bs 512 -nn 25
python ej_03_BN.py -lr 1e-2 -e 200 -bs 1024 -nn 25

# Ahora con 50 neuronas
python ej_03_BN.py -lr 1e-5 -e 200 -bs 64  -nn 50
python ej_03_BN.py -lr 1e-5 -e 200 -bs 128 -nn 50
python ej_03_BN.py -lr 1e-5 -e 200 -bs 256 -nn 50
python ej_03_BN.py -lr 1e-5 -e 200 -bs 512 -nn 50
python ej_03_BN.py -lr 1e-5 -e 200 -bs 1024 -nn 50

python ej_03_BN.py -lr 1e-4 -e 200 -bs 64  -nn 50
python ej_03_BN.py -lr 1e-4 -e 200 -bs 128 -nn 50
python ej_03_BN.py -lr 1e-4 -e 200 -bs 256 -nn 50
python ej_03_BN.py -lr 1e-4 -e 200 -bs 512 -nn 50
python ej_03_BN.py -lr 1e-4 -e 200 -bs 1024 -nn 50

python ej_03_BN.py -lr 1e-3 -e 200 -bs 64  -nn 50
python ej_03_BN.py -lr 1e-3 -e 200 -bs 128 -nn 50
python ej_03_BN.py -lr 1e-3 -e 200 -bs 256 -nn 50
python ej_03_BN.py -lr 1e-3 -e 200 -bs 512 -nn 50
python ej_03_BN.py -lr 1e-3 -e 200 -bs 1024 -nn 50

python ej_03_BN.py -lr 1e-2 -e 200 -bs 64  -nn 50
python ej_03_BN.py -lr 1e-2 -e 200 -bs 128 -nn 50
python ej_03_BN.py -lr 1e-2 -e 200 -bs 256 -nn 50
python ej_03_BN.py -lr 1e-2 -e 200 -bs 512 -nn 50
python ej_03_BN.py -lr 1e-2 -e 200 -bs 1024 -nn 50
