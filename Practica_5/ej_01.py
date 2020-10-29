#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
date: 28-10-2020
File: ej_01.py
Author : Facundo Martin Cabrera
Email: cabre94@hotmail.com facundo.cabrera@ib.edu.ar
GitHub: https://github.com/cabre94
GitLab: https://gitlab.com/cabre94
Description: ssh facundo.cabrera@rocks7frontend.fisica.cabib
"""

import os
import numpy as np
import matplotlib.pyplot as plt

import tensorflow.keras as keras
from tensorflow.keras import layers, activations, optimizers
from tensorflow.keras import regularizers

from tensorflow.keras.preprocessing.image import load_img, img_to_array

small_dataset = False
#path_data = "/home/cabre1994/Desktop/Deep_Learning/Deep_Learning_IB/Datasets"
path_data = os.getcwd()

if small_dataset:
    save_images = os.path.join(path_data, "dogs.vs-cats_small.npy")
    save_label = os.path.join(path_data, "dogs.vs-cats_small_label.npy")
    path_data = os.path.join(path_data, "dogs-vs-cats_small")
else:
    save_images = os.path.join(path_data, "dogs.vs-cats.npy")
    save_label = os.path.join(path_data, "dogs.vs-cats_label.npy")
    path_data = os.path.join(path_data, "dogs-vs-cats")

i = 1

if not os.path.exists(save_images):
    images = np.array([],dtype=np.uint8)
    labels = np.array([],dtype=np.uint8)

    for file in os.listdir(path_data):
        print(i)
        i += 1
        #if i > 100:
        #    continue
        
        if small_dataset:
            img = load_img(os.path.join(path_data,file))
        else:
            img = load_img(os.path.join(path_data,file), target_size=(299,299,3),interpolation="bilinear")

        img_arr = img_to_array(img).astype(np.uint8)

        images = np.append(images, img_arr)

        if file.startswith('cat'):
            labels = np.append(labels, 0)
        else:
            labels = np.append(labels, 1)
        #import ipdb; ipdb.set_trace(context=15)  # XXX BREAKPOINT

    if small_dataset:
        images = images.reshape(-1,32,32,3)
        np.save(save_images, images)
        np.save(save_label, labels)
    else:
        images = images.reshape(-1,299,299,3)
        np.save(save_images, images)
        np.save(save_label, labels)
