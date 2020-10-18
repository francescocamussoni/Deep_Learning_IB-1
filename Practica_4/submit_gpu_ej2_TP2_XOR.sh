#! /bin/bash
#$ -N output_2_TP2_XOR
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

python ej_02_XOR_a.py -lr 1e-5 -e 10000
python ej_02_XOR_a.py -lr 1e-4 -e 10000
python ej_02_XOR_a.py -lr 1e-3 -e 10000
python ej_02_XOR_a.py -lr 1e-2 -e 10000
python ej_02_XOR_a.py -lr 1e-1 -e 10000
python ej_02_XOR_a.py -lr 1e0  -e 10000

python ej_02_XOR_b.py -lr 1e-5 -e 10000
python ej_02_XOR_b.py -lr 1e-4 -e 10000
python ej_02_XOR_b.py -lr 1e-3 -e 10000
python ej_02_XOR_b.py -lr 1e-2 -e 10000
python ej_02_XOR_b.py -lr 1e-1 -e 10000
python ej_02_XOR_b.py -lr 1e0  -e 10000