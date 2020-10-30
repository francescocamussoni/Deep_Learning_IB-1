#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
date: 17-10-2020
File: utils.py
Author : Facundo Martin Cabrera
Email: cabre94@hotmail.com facundo.cabrera@ib.edu.ar
GitHub: https://github.com/cabre94
GitLab: https://gitlab.com/cabre94
Description: .py auxiliar para pasar argumentos por linea de comandos
"""

import argparse

# Argumentos por linea de comandos
parser = argparse.ArgumentParser()
parser.add_argument(
    "-lr",
    "--learning_rate",
    type=float,
    default=1e-3,
    help="Learning rate (default: 1e-3)",
)
parser.add_argument(
    "-rf",
    "--regularizer_factor",
    type=float,
    default=0,
    help="Regularizer factor (default: 0)",
)
parser.add_argument(
    "-e",
    "--epochs",
    type=int,
    default=200,
    help="Epochs (default: 200)",
)
parser.add_argument(
    "-bs",
    "--batch_size",
    type=int,
    default=None,
    help="Batch size (default: None)",
)
parser.add_argument(
    "-do",
    "--Dropout",
    type=float,
    default=0,
    help="Dropout argument (default: 0)",
)
parser.add_argument(
    "-nn",
    "--NumNeuronas",
    type=int,
    default=10,
    help="Numero de neuronas (default: 10)",
)
parser.add_argument(
    "-ed",
    "--EmbeddingDim",
    type=int,
    default=100,
    help="Dimension de salida del embedding (default: 100)",
)
parser.add_argument(
    # "-lr",
    "--dataset",
    type=str,
    default='cifar10',
    help="Dataset (default: cifar10)",
)
parser.add_argument(
    "--small",
    type=int,
    default=1,
    help="Small dataset (default: True)",
)
kwargs = vars(parser.parse_args())
lr = kwargs["learning_rate"]
rf = kwargs["regularizer_factor"]
epochs = kwargs['epochs']
batch_size = kwargs['batch_size']
drop_arg = kwargs['Dropout']
nn = kwargs['NumNeuronas']
embedding_dim = kwargs['EmbeddingDim']
dataset = kwargs["dataset"]
small_dataset = kwargs['small']

description = 'lr={:.1e}_rf={:.1e}_do={}_epochs={}_bs={}_nn={}_ed={}'.format(
    lr, rf, drop_arg, epochs, batch_size, nn, embedding_dim)
print(small_dataset)    
print("\n-------------------------------------")
print('lr: {:.1e} rf: {:.1e} do: {} epochs: {} bs: {} nn: {} ed: {}'.format(
    lr, rf, drop_arg, epochs, batch_size, nn, embedding_dim))
print("-------------------------------------\n")
