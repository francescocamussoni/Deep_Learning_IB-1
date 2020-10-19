#! /bin/bash
#$ -N o_4_Embe
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
python ej_04_embeddings.py -lr 1e-5 -e 200 -ed 50  -bs 256 -nn 25
python ej_04_embeddings.py -lr 1e-5 -e 200 -ed 75  -bs 256 -nn 25
python ej_04_embeddings.py -lr 1e-5 -e 200 -ed 100 -bs 256 -nn 25
python ej_04_embeddings.py -lr 1e-5 -e 200 -ed 125 -bs 256 -nn 25
python ej_04_embeddings.py -lr 1e-5 -e 200 -ed 150 -bs 256 -nn 25

python ej_04_embeddings.py -lr 1e-4 -e 200 -ed 50  -bs 256 -nn 25
python ej_04_embeddings.py -lr 1e-4 -e 200 -ed 75  -bs 256 -nn 25
python ej_04_embeddings.py -lr 1e-4 -e 200 -ed 100 -bs 256 -nn 25
python ej_04_embeddings.py -lr 1e-4 -e 200 -ed 125 -bs 256 -nn 25
python ej_04_embeddings.py -lr 1e-4 -e 200 -ed 150 -bs 256 -nn 25

python ej_04_embeddings.py -lr 1e-3 -e 200 -ed 50  -bs 256 -nn 25
python ej_04_embeddings.py -lr 1e-3 -e 200 -ed 75  -bs 256 -nn 25
python ej_04_embeddings.py -lr 1e-3 -e 200 -ed 100 -bs 256 -nn 25
python ej_04_embeddings.py -lr 1e-3 -e 200 -ed 125 -bs 256 -nn 25
python ej_04_embeddings.py -lr 1e-3 -e 200 -ed 150 -bs 256 -nn 25

python ej_04_embeddings.py -lr 1e-2 -e 200 -ed 50  -bs 256 -nn 25
python ej_04_embeddings.py -lr 1e-2 -e 200 -ed 75  -bs 256 -nn 25
python ej_04_embeddings.py -lr 1e-2 -e 200 -ed 100 -bs 256 -nn 25
python ej_04_embeddings.py -lr 1e-2 -e 200 -ed 125 -bs 256 -nn 25
python ej_04_embeddings.py -lr 1e-2 -e 200 -ed 150 -bs 256 -nn 25

python ej_04_embeddings.py -lr 1e-1 -e 200 -ed 50  -bs 256 -nn 25
python ej_04_embeddings.py -lr 1e-1 -e 200 -ed 75  -bs 256 -nn 25
python ej_04_embeddings.py -lr 1e-1 -e 200 -ed 100 -bs 256 -nn 25
python ej_04_embeddings.py -lr 1e-1 -e 200 -ed 125 -bs 256 -nn 25
python ej_04_embeddings.py -lr 1e-1 -e 200 -ed 150 -bs 256 -nn 25

