#! /bin/bash
#$ -N o_4_Conv
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
# python ej_04_Conv1D.py -lr 1e-6 -e 20 -bs 256 -ed 50 -do 0.2 # Parecen dar mal
# python ej_04_Conv1D.py -lr 1e-6 -e 20 -bs 256 -ed 50 -do 0.4
# python ej_04_Conv1D.py -lr 1e-6 -e 20 -bs 256 -ed 50 -do 0.6
# python ej_04_Conv1D.py -lr 1e-6 -e 20 -bs 256 -ed 50 -do 0.8

python ej_04_Conv1D.py -lr 1e-1 -e 20 -bs 256 -ed 50 -do 0.2
python ej_04_Conv1D.py -lr 1e-1 -e 20 -bs 256 -ed 50 -do 0.4
python ej_04_Conv1D.py -lr 1e-1 -e 20 -bs 256 -ed 50 -do 0.6
python ej_04_Conv1D.py -lr 1e-1 -e 20 -bs 256 -ed 50 -do 0.8

python ej_04_Conv1D.py -lr 1e-2 -e 20 -bs 256 -ed 50 -do 0.2
python ej_04_Conv1D.py -lr 1e-2 -e 20 -bs 256 -ed 50 -do 0.4
python ej_04_Conv1D.py -lr 1e-2 -e 20 -bs 256 -ed 50 -do 0.6
python ej_04_Conv1D.py -lr 1e-2 -e 20 -bs 256 -ed 50 -do 0.8

python ej_04_Conv1D.py -lr 1e-3 -e 20 -bs 256 -ed 50 -do 0.2
python ej_04_Conv1D.py -lr 1e-3 -e 20 -bs 256 -ed 50 -do 0.4
python ej_04_Conv1D.py -lr 1e-3 -e 20 -bs 256 -ed 50 -do 0.6
python ej_04_Conv1D.py -lr 1e-3 -e 20 -bs 256 -ed 50 -do 0.8

python ej_04_Conv1D.py -lr 1e-4 -e 20 -bs 256 -ed 50 -do 0.2
python ej_04_Conv1D.py -lr 1e-4 -e 20 -bs 256 -ed 50 -do 0.4
python ej_04_Conv1D.py -lr 1e-4 -e 20 -bs 256 -ed 50 -do 0.6
python ej_04_Conv1D.py -lr 1e-4 -e 20 -bs 256 -ed 50 -do 0.8

python ej_04_Conv1D.py -lr 1e-5 -e 20 -bs 256 -ed 50 -do 0.2
python ej_04_Conv1D.py -lr 1e-5 -e 20 -bs 256 -ed 50 -do 0.4
python ej_04_Conv1D.py -lr 1e-5 -e 20 -bs 256 -ed 50 -do 0.6
python ej_04_Conv1D.py -lr 1e-5 -e 20 -bs 256 -ed 50 -do 0.8
