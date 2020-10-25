#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
date: 24-10-2020
File: graphics.py
Author : Facundo Martin Cabrera
Email: cabre94@hotmail.com facundo.cabrera@ib.edu.ar
GitHub: https://github.com/cabre94
GitLab: https://gitlab.com/cabre94
Description:
"""

import os
import numpy as np
from matplotlib import pyplot as plt
from tensorflow import keras

import seaborn as snn
snn.set(font_scale = 1)
snn.set_style("darkgrid", {"axes.facecolor": ".9"})







def sinplot(flip=1):
    x = np.linspace(0, 14, 100)
    for i in range(1, 7):
        plt.plot(x, np.sin(x + i * .5) * (7 - i) * flip)
    plt.show()



# sinplot()





def graficos_2_EJ3TP2():
    path_data  = os.path.join("Informe","Datos","2_EJ3_TP2","lr=0.0001_rf=0.001_do=1_epochs=300_bs=128_nn=10.npy")
    path_model  = os.path.join("Informe","Datos","2_EJ3_TP2","lr=0.0001_rf=0.001_do=1_epochs=300_bs=128_nn=10.h5")

    data = np.load(path_data,allow_pickle=True).item()
    model = keras.models.load_model(path_model)

    # import ipdb; ipdb.set_trace(context=15)  # XXX BREAKPOINT

   # Grafico y guardo figuras
    img_folder = os.path.join('Informe','Figuras', '2_EJ3_TP2')
    if not os.path.exists(img_folder):
        os.makedirs(img_folder)

    # Grafico
    plt.plot(data['loss'], label="Loss Training")
    plt.plot(data['val_loss'], label="Loss Validation")
    # plt.title("Acc Test: {:.3f}".format(test_Acc))
    plt.xlabel("Epocas")
    plt.ylabel("Loss")
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(img_folder, 'Loss.pdf'),
                format="pdf",
                bbox_inches="tight")
    plt.show()

    plt.plot(data['Acc'], label="Acc. Training")
    plt.plot(data['val_Acc'], label="Acc. Validation")
    # plt.title("Acc Test: {:.3f}".format(test_Acc))
    plt.xlabel("Epocas")
    plt.ylabel("Accuracy")
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(img_folder, 'Acc.pdf'),
                format="pdf",
                bbox_inches="tight")
    plt.show()

def graficos_2_EJ4TP2():
    path_data  = os.path.join("Informe","Datos","2_EJ4_TP2","lr=0.001_rf=0.001_do=1_epochs=300_bs=128_nn=10.npy")
    path_model  = os.path.join("Informe","Datos","2_EJ4_TP2","lr=0.001_rf=0.001_do=1_epochs=300_bs=128_nn=10.h5")

    data = np.load(path_data,allow_pickle=True).item()
    model = keras.models.load_model(path_model)

    # import ipdb; ipdb.set_trace(context=15)  # XXX BREAKPOINT

   # Grafico y guardo figuras
    img_folder = os.path.join('Informe','Figuras', '2_EJ4_TP2')
    if not os.path.exists(img_folder):
        os.makedirs(img_folder)

    # Grafico
    plt.plot(data['loss'], label="Loss Training")
    plt.plot(data['val_loss'], label="Loss Validation")
    # plt.title("Acc Test: {:.3f}".format(test_Acc))
    plt.xlabel("Epocas")
    plt.ylabel("Loss")
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(img_folder, 'Loss.pdf'),
                format="pdf",
                bbox_inches="tight")
    plt.show()

    plt.plot(data['Acc'], label="Acc. Training")
    plt.plot(data['val_Acc'], label="Acc. Validation")
    # plt.title("Acc Test: {:.3f}".format(test_Acc))
    plt.xlabel("Epocas")
    plt.ylabel("Accuracy")
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(img_folder, 'Acc.pdf'),
                format="pdf",
                bbox_inches="tight")
    plt.show()

def graficos_2_XOR():
    path_data_a  = os.path.join("Informe","Datos","2_XOR_A","lr=0.001_rf=0_do=1_epochs=10000_bs=None_nn=10.npy")
    path_model_a  = os.path.join("Informe","Datos","2_XOR_A","lr=0.001_rf=0_do=1_epochs=10000_bs=None_nn=10.h5")

    path_data_b  = os.path.join("Informe","Datos","2_XOR_B","lr=1.0e-03_rf=0.0e+00_do=0_epochs=10000_bs=None_nn=10_ed=100.npy")
    path_model_b  = os.path.join("Informe","Datos","2_XOR_B","lr=1.0e-03_rf=0.0e+00_do=0_epochs=10000_bs=None_nn=10_ed=100.h5")

    data_a = np.load(path_data_a,allow_pickle=True).item()
    # model_a = keras.models.load_model(path_model_a)

    data_b = np.load(path_data_b,allow_pickle=True).item()
    # model_b = keras.models.load_model(path_model_b)

   # Grafico y guardo figuras
    img_folder = os.path.join('Informe','Figuras', '2_XOR')
    if not os.path.exists(img_folder):
        os.makedirs(img_folder)

    # Grafico
    plt.plot(data_a['loss'], label="1° Arq.")
    plt.plot(data_b['loss'], label="2° Arq.")
    plt.xlabel("Epocas")
    plt.ylabel("Loss")
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(img_folder, 'Loss.pdf'),
                format="pdf",
                bbox_inches="tight")
    plt.show()

    plt.plot(data_a['my_acc'], label="1° Arq.")
    plt.plot(data_b['my_acc'], label="2° Arq.")
    plt.xlabel("Epocas")
    plt.ylabel("Accuracy")
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(img_folder, 'Acc.pdf'),
                format="pdf",
                bbox_inches="tight")
    plt.show()

def graficos_3_BN():
    path_data  = os.path.join("Informe","Datos","3_BN","lr=0.0001_rf=0_do=1_epochs=200_bs=128_nn=25.npy")
    path_model  = os.path.join("Informe","Datos","3_BN","lr=0.0001_rf=0_do=1_epochs=200_bs=128_nn=25.h5")

    data = np.load(path_data,allow_pickle=True).item()
    model = keras.models.load_model(path_model)

   # Grafico y guardo figuras
    img_folder = os.path.join('Informe','Figuras', '3_BN')
    if not os.path.exists(img_folder):
        os.makedirs(img_folder)

    # Grafico
    plt.plot(data['loss'][:100], label="Loss Training")
    plt.plot(data['val_loss'][:100], label="Loss Validation")
    # plt.title("Acc Test: {:.3f}".format(test_Acc))
    plt.xlabel("Epocas")
    plt.ylabel("Loss")
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(img_folder, 'Loss.pdf'),
                format="pdf",
                bbox_inches="tight")
    plt.show()

    plt.plot(data['B_Acc'][:100], label="Acc. Training")
    plt.plot(data['val_B_Acc'][:100], label="Acc. Validation")
    # plt.title("Acc Test: {:.3f}".format(test_Acc))
    plt.xlabel("Epocas")
    plt.ylabel("Accuracy")
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(img_folder, 'Acc.pdf'),
                format="pdf",
                bbox_inches="tight")
    plt.show()

def graficos_3_Dropout():
    path_data  = os.path.join("Informe","Datos","3_Dropout","lr=0.0001_rf=0.0_do=0.75_epochs=200_bs=64_nn=25_ed=100.npy")
    path_model  = os.path.join("Informe","Datos","3_Dropout","lr=0.0001_rf=0.0_do=0.75_epochs=200_bs=64_nn=25_ed=100.h5")

    data = np.load(path_data,allow_pickle=True).item()
    model = keras.models.load_model(path_model)

   # Grafico y guardo figuras
    img_folder = os.path.join('Informe','Figuras', '3_Dropout')
    if not os.path.exists(img_folder):
        os.makedirs(img_folder)

    # Grafico
    plt.plot(data['loss'][:100], label="Loss Training")
    plt.plot(data['val_loss'][:100], label="Loss Validation")
    # plt.title("Acc Test: {:.3f}".format(test_Acc))
    plt.xlabel("Epocas")
    plt.ylabel("Loss")
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(img_folder, 'Loss.pdf'),
                format="pdf",
                bbox_inches="tight")
    plt.show()

    plt.plot(data['B_Acc'][:100], label="Acc. Training")
    plt.plot(data['val_B_Acc'][:100], label="Acc. Validation")
    # plt.title("Acc Test: {:.3f}".format(test_Acc))
    plt.xlabel("Epocas")
    plt.ylabel("Accuracy")
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(img_folder, 'Acc.pdf'),
                format="pdf",
                bbox_inches="tight")
    plt.show()

def graficos_3_L2():
    path_data  = os.path.join("Informe","Datos","3_L2","lr=0.0001_rf=0.1_do=0_epochs=200_bs=64_nn=25_ed=100.npy")
    path_model  = os.path.join("Informe","Datos","3_L2","lr=0.0001_rf=0.1_do=0_epochs=200_bs=64_nn=25_ed=100.h5")

    data = np.load(path_data,allow_pickle=True).item()
    model = keras.models.load_model(path_model)

   # Grafico y guardo figuras
    img_folder = os.path.join('Informe','Figuras', '3_L2')
    if not os.path.exists(img_folder):
        os.makedirs(img_folder)

    # Grafico
    plt.plot(data['loss'][:100], label="Loss Training")
    plt.plot(data['val_loss'][:100], label="Loss Validation")
    # plt.title("Acc Test: {:.3f}".format(test_Acc))
    plt.xlabel("Epocas")
    plt.ylabel("Loss")
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(img_folder, 'Loss.pdf'),
                format="pdf",
                bbox_inches="tight")
    plt.show()

    plt.plot(data['B_Acc'][:100], label="Acc. Training")
    plt.plot(data['val_B_Acc'][:100], label="Acc. Validation")
    # plt.title("Acc Test: {:.3f}".format(test_Acc))
    plt.xlabel("Epocas")
    plt.ylabel("Accuracy")
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(img_folder, 'Acc.pdf'),
                format="pdf",
                bbox_inches="tight")
    plt.show()

def graficos_4_Embedding():
    path_data  = os.path.join("Informe","Datos","4_Embedding","lr=1.0e-05_rf=0_do=0_epochs=200_bs=256_nn=25_ed=75.npy")
    path_model  = os.path.join("Informe","Datos","4_Embedding","lr=1.0e-05_rf=0_do=0_epochs=200_bs=256_nn=25_ed=75.h5")

    data = np.load(path_data,allow_pickle=True).item()
    model = keras.models.load_model(path_model)

   # Grafico y guardo figuras
    img_folder = os.path.join('Informe','Figuras', '4_Embedding')
    if not os.path.exists(img_folder):
        os.makedirs(img_folder)

    # Grafico
    plt.plot(data['loss'][:100], label="Loss Training")
    plt.plot(data['val_loss'][:100], label="Loss Validation")
    # plt.title("Acc Test: {:.3f}".format(test_Acc))
    plt.xlabel("Epocas")
    plt.ylabel("Loss")
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(img_folder, 'Loss.pdf'),
                format="pdf",
                bbox_inches="tight")
    plt.show()

    plt.plot(data['B_Acc'][:100], label="Acc. Training")
    plt.plot(data['val_B_Acc'][:100], label="Acc. Validation")
    # plt.title("Acc Test: {:.3f}".format(test_Acc))
    plt.xlabel("Epocas")
    plt.ylabel("Accuracy")
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(img_folder, 'Acc.pdf'),
                format="pdf",
                bbox_inches="tight")
    plt.show()

def graficos_4_Conv():
    path_data  = os.path.join("Informe","Datos","4_Conv","lr=1.0e-05_rf=0_do=0.2_epochs=200_bs=256_nn=10_ed=50.npy")
    path_model  = os.path.join("Informe","Datos","4_Conv","lr=1.0e-05_rf=0_do=0.2_epochs=200_bs=256_nn=10_ed=50.h5")

    data = np.load(path_data,allow_pickle=True).item()
    model = keras.models.load_model(path_model)

   # Grafico y guardo figuras
    img_folder = os.path.join('Informe','Figuras', '4_Conv')
    if not os.path.exists(img_folder):
        os.makedirs(img_folder)

    # Grafico
    plt.plot(data['loss'], label="Loss Training")
    plt.plot(data['val_loss'], label="Loss Validation")
    # plt.title("Acc Test: {:.3f}".format(test_Acc))
    plt.xlabel("Epocas")
    plt.ylabel("Loss")
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(img_folder, 'Loss.pdf'),
                format="pdf",
                bbox_inches="tight")
    plt.show()

    plt.plot(data['B_Acc'], label="Acc. Training")
    plt.plot(data['val_B_Acc'], label="Acc. Validation")
    # plt.title("Acc Test: {:.3f}".format(test_Acc))
    plt.xlabel("Epocas")
    plt.ylabel("Accuracy")
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(img_folder, 'Acc.pdf'),
                format="pdf",
                bbox_inches="tight")
    plt.show()

def mapeoLogistico(x):
    return 4*x*(1-x)

def graficos_5():
    path_data  = os.path.join("Informe","Datos","5","lr=1.0e-03_rf=0.0_do=0_epochs=200_bs=256_nn=10_ed=100.npy")
    path_model  = os.path.join("Informe","Datos","5","lr=1.0e-03_rf=0.0_do=0_epochs=200_bs=256_nn=10_ed=100.h5")

    data = np.load(path_data,allow_pickle=True).item()
    model = keras.models.load_model(path_model)

   # Grafico y guardo figuras
    img_folder = os.path.join('Informe','Figuras', '5')
    if not os.path.exists(img_folder):
        os.makedirs(img_folder)

    # Grafico
    plt.plot(data['loss'], label="Loss Training")
    plt.plot(data['val_loss'], label="Loss Validation")
    # plt.title("Acc Test: {:.3f}".format(test_Acc))
    plt.xlabel("Epocas")
    plt.ylabel("Loss")
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(img_folder, 'Loss.pdf'),
                format="pdf",
                bbox_inches="tight")
    plt.show()

    np.random.seed(10)
    x_0 = np.random.rand()

    exact = np.array([x_0])
    wNN = np.array([x_0])

    for i in range(100):
        x_t = mapeoLogistico(exact[-1])
        x_t_nn = model.predict([wNN[-1]])

        exact = np.append(exact, x_t)
        wNN = np.append(wNN, x_t_nn)
    
    plt.plot(exact, label="Exacta")
    plt.plot(wNN, label="Con RN")
    plt.xlabel("Iteracioines")
    plt.ylabel(r"$x(t)$")
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.11), ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(img_folder, 'Evolucion.pdf'),
                format="pdf",
                bbox_inches="tight")
    plt.show()


    




if __name__ == "__main__":
    # graficos_2_EJ3TP2()

    # graficos_2_EJ4TP2()

    # graficos_2_XOR()

    # graficos_3_BN()

    # graficos_3_Dropout()

    # graficos_3_L2()

    # graficos_4_Embedding()

    # graficos_4_Conv()

    # graficos_5()

    



