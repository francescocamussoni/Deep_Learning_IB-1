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

python ej_04_Embeddings.py -lr 1e-5 -e 200 -ed 25 -bs 256 -nn 25			# Mucho overfitting 0.87
python ej_04_Embeddings.py -lr 1e-5 -e 200 -ed 50 -bs 256 -nn 25			# Mas epocas
python ej_04_Embeddings.py -lr 1e-5 -e 200 -ed 75 -bs 256 -nn 25		# Mas epocas
#python ej_04_Embeddings.py -lr 1e-5 -e 10 -ed 100 -bs 64 -nn 25		# Da mal
#python ej_04_Embeddings.py -lr 1e-5 -e 10 -ed 125 -bs 64 -nn 25		# Da mal

python ej_04_Embeddings.py -lr 1e-4 -e 200 -ed 25 -bs 256 -nn 25			# Mas epocas, el mejorcito
python ej_04_Embeddings.py -lr 1e-4 -e 200 -ed 50 -bs 256 -nn 25			# Mas epocas, el mejorcito
python ej_04_Embeddings.py -lr 1e-4 -e 200 -ed 75 -bs 256 -nn 25		# Mas epocas
#python ej_04_Embeddings.py -lr 1e-4 -e 10 -ed 100 -bs 64 -nn 25		# Mas epocas
#python ej_04_Embeddings.py -lr 1e-4 -e 10 -ed 125 -bs 64 -nn 25		# Overfitting

python ej_04_Embeddings.py -lr 1e-3 -e 200 -ed 25 -bs 256 -nn 25			# Overfitting
python ej_04_Embeddings.py -lr 1e-3 -e 200 -ed 50 -bs 256 -nn 25		# Overfitting
python ej_04_Embeddings.py -lr 1e-3 -e 200 -ed 75 -bs 256 -nn 25		# Overfitting
#python ej_04_Embeddings.py -lr 1e-3 -e 10 -ed 100 -bs 64 -nn 25		# Overffiting
#python ej_04_Embeddings.py -lr 1e-3 -e 10 -ed 125 -bs 64 -nn 25		# Mucho overfitting

#python ej_04_Embeddings.py -lr 1e-2 -e 10 -ed 25 -bs 64 -nn 25			# Raro
#python ej_04_Embeddings.py -lr 1e-2 -e 10 -ed 50  -bs 64 -nn 25		# Overfitting
#python ej_04_Embeddings.py -lr 1e-2 -e 10 -ed 75  -bs 64 -nn 25		# Overfitting
#python ej_04_Embeddings.py -lr 1e-2 -e 10 -ed 100 -bs 64 -nn 25		# Overfitting
#python ej_04_Embeddings.py -lr 1e-2 -e 10 -ed 125 -bs 64 -nn 25		# Da mal

#python ej_04_Embeddings.py -lr 1e-1 -e 10 -ed 25 -bs 64 -nn 25			# Da mal
#python ej_04_Embeddings.py -lr 1e-1 -e 10 -ed 50  -bs 64 -nn 25		# Da mal
#python ej_04_Embeddings.py -lr 1e-1 -e 10 -ed 75  -bs 64 -nn 25		# Da mal
#python ej_04_Embeddings.py -lr 1e-1 -e 10 -ed 100 -bs 64 -nn 25		# Da mal
#python ej_04_Embeddings.py -lr 1e-1 -e 10 -ed 125 -bs 64 -nn 25		# Da mal

