#! /bin/bash
#$ -N o_2_TP2_EJ3
#$ -cwd
#$ -j y
#$ -S /bin/bash
#$ -V
##  pido la cola gpu.q
#$ -q gpu@compute-6-5.local
## pido una placa
#$ -l gpu=1
#$ -l memoria_a_usar=1G
#
# Load gpu drivers and conda
module load miniconda

source activate deep_learning

# Execute the script
hostname

python ej_02_3.py -lr 1e-5 -rf 1e-1 -e 300 -bs 128
python ej_02_3.py -lr 1e-5 -rf 1e-2 -e 300 -bs 128
python ej_02_3.py -lr 1e-5 -rf 1e-3 -e 300 -bs 128
python ej_02_3.py -lr 1e-5 -rf 0    -e 300 -bs 128

python ej_02_3.py -lr 1e-4 -rf 1e-1 -e 300 -bs 128
python ej_02_3.py -lr 1e-4 -rf 1e-2 -e 300 -bs 128
python ej_02_3.py -lr 1e-4 -rf 1e-3 -e 300 -bs 128
python ej_02_3.py -lr 1e-4 -rf 0    -e 300 -bs 128

python ej_02_3.py -lr 1e-3 -rf 1e-1 -e 300 -bs 128
python ej_02_3.py -lr 1e-3 -rf 1e-2 -e 300 -bs 128
python ej_02_3.py -lr 1e-3 -rf 1e-3 -e 300 -bs 128
python ej_02_3.py -lr 1e-3 -rf 0    -e 300 -bs 128

python ej_02_3.py -lr 1e-2 -rf 1e-1 -e 300 -bs 128
python ej_02_3.py -lr 1e-2 -rf 1e-2 -e 300 -bs 128
python ej_02_3.py -lr 1e-2 -rf 1e-3 -e 300 -bs 128
python ej_02_3.py -lr 1e-2 -rf 0    -e 300 -bs 128

python ej_02_3.py -lr 1e-1 -rf 1e-1 -e 300 -bs 128
python ej_02_3.py -lr 1e-1 -rf 1e-2 -e 300 -bs 128
python ej_02_3.py -lr 1e-1 -rf 1e-3 -e 300 -bs 128
python ej_02_3.py -lr 1e-1 -rf 0    -e 300 -bs 128