#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
date: 22-10-2020
File: Cabrera.py
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
snn.set(font_scale=1)
snn.set_style("darkgrid", {"axes.facecolor": ".9"})


def graficos_2_EJ3TP2():
    path_data = os.path.join(
        "Informe", "Datos", "2_EJ3_TP2",
        "lr=0.0001_rf=0.001_do=1_epochs=300_bs=128_nn=10.npy")
    path_model = os.path.join(
        "Informe", "Datos", "2_EJ3_TP2",
        "lr=0.0001_rf=0.001_do=1_epochs=300_bs=128_nn=10.h5")

    data = np.load(path_data, allow_pickle=True).item()
    model = keras.models.load_model(path_model)

    # import ipdb; ipdb.set_trace(context=15)  # XXX BREAKPOINT

    # Grafico y guardo figuras
    img_folder = os.path.join('Informe', 'Figuras', '2_EJ3_TP2')
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
    path_data = os.path.join(
        "Informe", "Datos", "2_EJ4_TP2",
        "lr=0.001_rf=0.001_do=1_epochs=300_bs=128_nn=10.npy")
    path_model = os.path.join(
        "Informe", "Datos", "2_EJ4_TP2",
        "lr=0.001_rf=0.001_do=1_epochs=300_bs=128_nn=10.h5")

    data = np.load(path_data, allow_pickle=True).item()
    model = keras.models.load_model(path_model)

    # import ipdb; ipdb.set_trace(context=15)  # XXX BREAKPOINT

    # Grafico y guardo figuras
    img_folder = os.path.join('Informe', 'Figuras', '2_EJ4_TP2')
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
    path_data_a = os.path.join(
        "Informe", "Datos", "2_XOR_A",
        "lr=0.001_rf=0_do=1_epochs=10000_bs=None_nn=10.npy")
    path_model_a = os.path.join(
        "Informe", "Datos", "2_XOR_A",
        "lr=0.001_rf=0_do=1_epochs=10000_bs=None_nn=10.h5")

    path_data_b = os.path.join(
        "Informe", "Datos", "2_XOR_B",
        "lr=1.0e-03_rf=0.0e+00_do=0_epochs=10000_bs=None_nn=10_ed=100.npy")
    path_model_b = os.path.join(
        "Informe", "Datos", "2_XOR_B",
        "lr=1.0e-03_rf=0.0e+00_do=0_epochs=10000_bs=None_nn=10_ed=100.h5")

    data_a = np.load(path_data_a, allow_pickle=True).item()
    # model_a = keras.models.load_model(path_model_a)

    data_b = np.load(path_data_b, allow_pickle=True).item()
    # model_b = keras.models.load_model(path_model_b)

    # Grafico y guardo figuras
    img_folder = os.path.join('Informe', 'Figuras', '2_XOR')
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
    path_data = os.path.join(
        "Informe", "Datos", "3_BN",
        "lr=0.0001_rf=0_do=1_epochs=200_bs=128_nn=25.npy")
    path_model = os.path.join(
        "Informe", "Datos", "3_BN",
        "lr=0.0001_rf=0_do=1_epochs=200_bs=128_nn=25.h5")

    data = np.load(path_data, allow_pickle=True).item()
    model = keras.models.load_model(path_model)

    # Grafico y guardo figuras
    img_folder = os.path.join('Informe', 'Figuras', '3_BN')
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
    path_data = os.path.join(
        "Informe", "Datos", "3_Dropout",
        "lr=0.0001_rf=0.0_do=0.75_epochs=200_bs=64_nn=25_ed=100.npy")
    path_model = os.path.join(
        "Informe", "Datos", "3_Dropout",
        "lr=0.0001_rf=0.0_do=0.75_epochs=200_bs=64_nn=25_ed=100.h5")

    data = np.load(path_data, allow_pickle=True).item()
    model = keras.models.load_model(path_model)

    # Grafico y guardo figuras
    img_folder = os.path.join('Informe', 'Figuras', '3_Dropout')
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
    path_data = os.path.join(
        "Informe", "Datos", "3_L2",
        "lr=0.0001_rf=0.1_do=0_epochs=200_bs=64_nn=25_ed=100.npy")
    path_model = os.path.join(
        "Informe", "Datos", "3_L2",
        "lr=0.0001_rf=0.1_do=0_epochs=200_bs=64_nn=25_ed=100.h5")

    data = np.load(path_data, allow_pickle=True).item()
    model = keras.models.load_model(path_model)

    # Grafico y guardo figuras
    img_folder = os.path.join('Informe', 'Figuras', '3_L2')
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
    path_data = os.path.join(
        "Informe", "Datos", "4_Embedding",
        "lr=1.0e-05_rf=0_do=0_epochs=200_bs=256_nn=25_ed=75.npy")
    path_model = os.path.join(
        "Informe", "Datos", "4_Embedding",
        "lr=1.0e-05_rf=0_do=0_epochs=200_bs=256_nn=25_ed=75.h5")

    data = np.load(path_data, allow_pickle=True).item()
    model = keras.models.load_model(path_model)

    # Grafico y guardo figuras
    img_folder = os.path.join('Informe', 'Figuras', '4_Embedding')
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
    path_data = os.path.join(
        "Informe", "Datos", "4_Conv",
        "lr=1.0e-05_rf=0_do=0.2_epochs=200_bs=256_nn=10_ed=50.npy")
    path_model = os.path.join(
        "Informe", "Datos", "4_Conv",
        "lr=1.0e-05_rf=0_do=0.2_epochs=200_bs=256_nn=10_ed=50.h5")

    data = np.load(path_data, allow_pickle=True).item()
    model = keras.models.load_model(path_model)

    # Grafico y guardo figuras
    img_folder = os.path.join('Informe', 'Figuras', '4_Conv')
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
    return 4 * x * (1 - x)


def graficos_5():
    path_data = os.path.join(
        "Informe", "Datos", "5",
        "lr=1.0e-03_rf=0.0_do=0_epochs=200_bs=256_nn=10_ed=100.npy")
    path_model = os.path.join(
        "Informe", "Datos", "5",
        "lr=1.0e-03_rf=0.0_do=0_epochs=200_bs=256_nn=10_ed=100.h5")

    data = np.load(path_data, allow_pickle=True).item()
    model = keras.models.load_model(path_model)

    # Grafico y guardo figuras
    img_folder = os.path.join('Informe', 'Figuras', '5')
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


def graficos_6():
    path_data = os.path.join(
        "Informe", "Datos", "6",
        "lr=1.0e-03_rf=1.0e-01_do=0_epochs=500_bs=32_nn=20_ed=100.npy")

    data = np.load(path_data, allow_pickle=True).item()

    # Grafico y guardo figuras
    img_folder = os.path.join('Informe', 'Figuras', '6')
    if not os.path.exists(img_folder):
        os.makedirs(img_folder)

    xx = np.arange(len(data['train_min']))
    plt.fill_between(xx, data['train_min'], data['train_max'], alpha=0.45)
    plt.plot(data['train_mean'])
    plt.xlabel("Epocas", fontsize=15)
    plt.ylabel("Accuracy", fontsize=15)
    # plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(img_folder, 'Acc.pdf'),
                format="pdf",
                bbox_inches="tight")
    plt.show()

    # Grafico
    plt.fill_between(xx, data['test_min'], data['test_max'], alpha=0.45)
    plt.plot(data['test_mean'])
    plt.xlabel("Epocas", fontsize=15)
    plt.ylabel("Test Accuracy", fontsize=15)
    plt.tight_layout()
    plt.savefig(os.path.join(img_folder, 'Acc_Test.pdf'),
                format="pdf",
                bbox_inches="tight")
    plt.show()

    plt.fill_between(xx, data['ltrain_min'], data['ltrain_max'], alpha=0.45)
    plt.plot(data['ltrain_mean'], label="Loss Training")
    plt.fill_between(xx, data['ltest_min'], data['ltest_max'], alpha=0.45)
    plt.plot(data['ltest_mean'], label="Loss Test")
    plt.xlabel("Epocas", fontsize=15)
    plt.ylabel("Loss", fontsize=15)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(img_folder, 'Loss.pdf'),
                format="pdf",
                bbox_inches="tight")
    plt.show()


def graficos_6_40N():
    path_data = os.path.join(
        "Informe", "Datos", "6",
        "lr=1.0e-03_rf=1.0e-01_do=0_epochs=500_bs=32_nn=40_ed=100.npy")

    data = np.load(path_data, allow_pickle=True).item()

    # Grafico y guardo figuras
    img_folder = os.path.join('Informe', 'Figuras', '6_40')
    if not os.path.exists(img_folder):
        os.makedirs(img_folder)

    xx = np.arange(len(data['train_min']))
    plt.fill_between(xx, data['train_min'], data['train_max'], alpha=0.45)
    plt.plot(data['train_mean'])
    plt.xlabel("Epocas", fontsize=15)
    plt.ylabel("Accuracy", fontsize=15)
    # plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(img_folder, 'Acc.pdf'),
                format="pdf",
                bbox_inches="tight")
    plt.show()

    # Grafico
    plt.fill_between(xx, data['test_min'], data['test_max'], alpha=0.45)
    plt.plot(data['test_mean'])
    plt.xlabel("Epocas", fontsize=15)
    plt.ylabel("Test Accuracy", fontsize=15)
    plt.tight_layout()
    plt.savefig(os.path.join(img_folder, 'Acc_Test.pdf'),
                format="pdf",
                bbox_inches="tight")
    plt.show()

    plt.fill_between(xx, data['ltrain_min'], data['ltrain_max'], alpha=0.45)
    plt.plot(data['ltrain_mean'], label="Loss Training")
    plt.fill_between(xx, data['ltest_min'], data['ltest_max'], alpha=0.45)
    plt.plot(data['ltest_mean'], label="Loss Test")
    plt.xlabel("Epocas", fontsize=15)
    plt.ylabel("Loss", fontsize=15)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(img_folder, 'Loss.pdf'),
                format="pdf",
                bbox_inches="tight")
    plt.show()


def graficos_6_CCorreccion():
    path_data = os.path.join(
        "Informe", "Datos", "6_Correccion",
        "lr=1.0e-03_rf=1.0e-01_do=0_epochs=500_bs=32_nn=40_ed=100.npy")

    data = np.load(path_data, allow_pickle=True).item()

    # Grafico y guardo figuras
    img_folder = os.path.join('Informe', 'Figuras', '6_Correccion')
    if not os.path.exists(img_folder):
        os.makedirs(img_folder)

    xx = np.arange(len(data['train_min']))
    plt.fill_between(xx, data['train_min'], data['train_max'], alpha=0.45)
    plt.plot(data['train_mean'])
    plt.xlabel("Epocas", fontsize=15)
    plt.ylabel("Accuracy", fontsize=15)
    # plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(img_folder, 'Acc.pdf'),
                format="pdf",
                bbox_inches="tight")
    plt.show()

    # Grafico
    plt.fill_between(xx, data['test_min'], data['test_max'], alpha=0.45)
    plt.plot(data['test_mean'])
    plt.xlabel("Epocas", fontsize=15)
    plt.ylabel("Test Accuracy", fontsize=15)
    plt.tight_layout()
    plt.savefig(os.path.join(img_folder, 'Acc_Test.pdf'),
                format="pdf",
                bbox_inches="tight")
    plt.show()

    plt.fill_between(xx, data['ltrain_min'], data['ltrain_max'], alpha=0.45)
    plt.plot(data['ltrain_mean'], label="Loss Training")
    plt.fill_between(xx, data['ltest_min'], data['ltest_max'], alpha=0.45)
    plt.plot(data['ltest_mean'], label="Loss Test")
    plt.xlabel("Epocas", fontsize=15)
    plt.ylabel("Loss", fontsize=15)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(img_folder, 'Loss.pdf'),
                format="pdf",
                bbox_inches="tight")
    plt.show()


def graficos_7():
    path_data = os.path.join(
        "Informe", "Datos", "7",
        "lr=1.0e-05_rf=0.0e+00_do=0_epochs=500_bs=256_nn=10_ed=100.npy")
    path_model = os.path.join(
        "Informe", "Datos", "7",
        "lr=1.0e-05_rf=0.0e+00_do=0_epochs=500_bs=256_nn=10_ed=100.h5")

    data = np.load(path_data, allow_pickle=True).item()
    model = keras.models.load_model(path_model)

    # Grafico y guardo figuras
    img_folder = os.path.join('Informe', 'Figuras', '7')
    if not os.path.exists(img_folder):
        os.makedirs(img_folder)

    # Grafico
    plt.plot(data['MSE'], label="Loss Training")
    plt.plot(data['val_MSE'], label="Loss Validation")
    # plt.title("Acc Test: {:.3f}".format(test_Acc))
    plt.xlabel("Epocas")
    plt.ylabel("Loss")
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(img_folder, 'Loss.pdf'),
                format="pdf",
                bbox_inches="tight")
    plt.show()


def graficos_7_v2():
    path_data = os.path.join(
        "Informe", "Datos", "7_Internet",
        "MSE_lr=1.0e-05_rf=0.0e+00_do=0_epochs=1000_bs=512_nn=10_ed=100.npy")
    path_model = os.path.join(
        "Informe", "Datos", "7_Internet",
        "MSE_lr=1.0e-05_rf=0.0e+00_do=0_epochs=1000_bs=512_nn=10_ed=100.h5")

    data = np.load(path_data, allow_pickle=True).item()
    model = keras.models.load_model(path_model)

    # Grafico y guardo figuras
    img_folder = os.path.join('Informe', 'Figuras', '7_Internet')
    if not os.path.exists(img_folder):
        os.makedirs(img_folder)

    # Grafico
    plt.plot(data['MSE'], label="Loss Training")
    plt.plot(data['val_MSE'], label="Loss Validation")
    # plt.title("Acc Test: {:.3f}".format(test_Acc))
    plt.xlabel("Epocas")
    plt.ylabel("Loss")
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(img_folder, 'Loss.pdf'),
                format="pdf",
                bbox_inches="tight")
    plt.show()


def graficos_8_Densa():
    path_data = os.path.join(
        "Informe", "Datos", "8_Dense",
        "lr=1.0e-04_rf=1.0e-03_do=0.4_epochs=200_bs=512_nn=10_ed=100.npy")
    path_model = os.path.join(
        "Informe", "Datos", "8_Dense",
        "lr=1.0e-04_rf=1.0e-03_do=0.4_epochs=200_bs=512_nn=10_ed=100.h5")

    data = np.load(path_data, allow_pickle=True).item()
    model = keras.models.load_model(path_model)

    # Grafico y guardo figuras
    img_folder = os.path.join('Informe', 'Figuras', '8_Dense')
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

    plt.plot(data['CAcc'][:100], label="Acc. Training")
    plt.plot(data['val_CAcc'][:100], label="Acc. Validation")
    # plt.title("Acc Test: {:.3f}".format(test_Acc))
    plt.xlabel("Epocas")
    plt.ylabel("Accuracy")
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(img_folder, 'Acc.pdf'),
                format="pdf",
                bbox_inches="tight")
    plt.show()


def graficos_8_Conv():
    path_data = os.path.join(
        "Informe", "Datos", "8_Conv",
        "lr=1.0e-04_rf=1.0e-03_do=0.4_epochs=200_bs=512_nn=10_ed=100.npy")
    path_model = os.path.join(
        "Informe", "Datos", "8_Conv",
        "lr=1.0e-04_rf=1.0e-03_do=0.4_epochs=200_bs=512_nn=10_ed=100.h5")

    data = np.load(path_data, allow_pickle=True).item()
    model = keras.models.load_model(path_model)

    # Grafico y guardo figuras
    img_folder = os.path.join('Informe', 'Figuras', '8_Conv')
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

    plt.plot(data['CAcc'][:100], label="Acc. Training")
    plt.plot(data['val_CAcc'][:100], label="Acc. Validation")
    # plt.title("Acc Test: {:.3f}".format(test_Acc))
    plt.xlabel("Epocas")
    plt.ylabel("Accuracy")
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(img_folder, 'Acc.pdf'),
                format="pdf",
                bbox_inches="tight")
    plt.show()


def graficos_9_Densa():
    path_data = os.path.join(
        "Informe", "Datos", "9_Dense",
        "lr=1.0e-04_rf=1.0e-03_do=0.4_epochs=200_bs=512_nn=10_ed=100.npy")
    path_model = os.path.join(
        "Informe", "Datos", "9_Dense",
        "lr=1.0e-04_rf=1.0e-03_do=0.4_epochs=200_bs=512_nn=10_ed=100.h5")

    data = np.load(path_data, allow_pickle=True).item()
    model = keras.models.load_model(path_model)

    # Grafico y guardo figuras
    img_folder = os.path.join('Informe', 'Figuras', '9_Dense')
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

    plt.plot(data['CAcc'][:100], label="Acc. Training")
    plt.plot(data['val_CAcc'][:100], label="Acc. Validation")
    # plt.title("Acc Test: {:.3f}".format(test_Acc))
    plt.xlabel("Epocas")
    plt.ylabel("Accuracy")
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(img_folder, 'Acc.pdf'),
                format="pdf",
                bbox_inches="tight")
    plt.show()


def graficos_9_Conv():
    path_data = os.path.join(
        "Informe", "Datos", "9_Conv",
        "lr=1.0e-04_rf=1.0e-03_do=0.4_epochs=200_bs=512_nn=10_ed=100.npy")
    path_model = os.path.join(
        "Informe", "Datos", "9_Conv",
        "lr=1.0e-04_rf=1.0e-03_do=0.4_epochs=200_bs=512_nn=10_ed=100.h5")

    data = np.load(path_data, allow_pickle=True).item()
    model = keras.models.load_model(path_model)

    # Grafico y guardo figuras
    img_folder = os.path.join('Informe', 'Figuras', '9_Conv')
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

    plt.plot(data['CAcc'][:100], label="Acc. Training")
    plt.plot(data['val_CAcc'][:100], label="Acc. Validation")
    # plt.title("Acc Test: {:.3f}".format(test_Acc))
    plt.xlabel("Epocas")
    plt.ylabel("Accuracy")
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(img_folder, 'Acc.pdf'),
                format="pdf",
                bbox_inches="tight")
    plt.show()


def graficos_10_AlexNet_Cifar10():
    path_data = os.path.join(
        "Informe", "Datos", "10_AlexNet_cifar10",
        "lr=1.0e-03_rf=1.0e-03_do=0_epochs=100_bs=64_nn=10_ed=100.npy")
    # path_model  = os.path.join("Informe","Datos","10_AlexNet_cifar10","lr=1.0e-03_rf=1.0e-03_do=0_epochs=100_bs=64_nn=10_ed=100.h5")

    data = np.load(path_data, allow_pickle=True).item()
    # model = keras.models.load_model(path_model)

    # Grafico y guardo figuras
    img_folder = os.path.join('Informe', 'Figuras', '10_AlexNet_cifar10')
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

    plt.plot(data['CAcc'], label="Acc. Training")
    plt.plot(data['val_CAcc'], label="Acc. Validation")
    # plt.title("Acc Test: {:.3f}".format(test_Acc))
    plt.xlabel("Epocas")
    plt.ylabel("Accuracy")
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(img_folder, 'Acc.pdf'),
                format="pdf",
                bbox_inches="tight")
    plt.show()


def graficos_10_AlexNet_Cifar100():
    path_data = os.path.join("Informe", "Datos", "10_AlexNet_cifar100",
                             "1.0e-03_3.0e-04_100_64_SINDROP.npy")
    # path_model  = os.path.join("Informe","Datos","10_AlexNet_cifar10","1.0e-03_3.0e-04_100_64_SINDROP.h5")

    data = np.load(path_data, allow_pickle=True).item()
    # model = keras.models.load_model(path_model)

    # Grafico y guardo figuras
    img_folder = os.path.join('Informe', 'Figuras', '10_AlexNet_cifar100')
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

    plt.plot(data['CAcc'], label="Acc. Training")
    plt.plot(data['val_CAcc'], label="Acc. Validation")
    # plt.title("Acc Test: {:.3f}".format(test_Acc))
    plt.xlabel("Epocas")
    plt.ylabel("Accuracy")
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(img_folder, 'Acc.pdf'),
                format="pdf",
                bbox_inches="tight")
    plt.show()


def graficos_10_VGG_Cifar10():
    path_data = os.path.join("Informe", "Datos", "10_VGG16_cifar10",
                             "1.0e-03_6.0e-04_50_256_SINDROP.npy")
    # path_model  = os.path.join("Informe","Datos","10_VGG16_cifar10","1.0e-03_6.0e-04_50_256_SINDROP.h5")

    data = np.load(path_data, allow_pickle=True).item()
    # model = keras.models.load_model(path_model)

    # Grafico y guardo figuras
    img_folder = os.path.join('Informe', 'Figuras', '10_VGG16_cifar10')
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

    plt.plot(data['CAcc'], label="Acc. Training")
    plt.plot(data['val_CAcc'], label="Acc. Validation")
    # plt.title("Acc Test: {:.3f}".format(test_Acc))
    plt.xlabel("Epocas")
    plt.ylabel("Accuracy")
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(img_folder, 'Acc.pdf'),
                format="pdf",
                bbox_inches="tight")
    plt.show()


def graficos_10_VGG_Cifar100():
    path_data = os.path.join("Informe", "Datos", "10_VGG16_cifar100",
                             "1.0e-03_6.0e-04_100_256_SINDROP.npy")
    # path_model  = os.path.join("Informe","Datos","10_VGG16_cifar10","1.0e-03_6.0e-04_100_256_SINDROP.h5")

    data = np.load(path_data, allow_pickle=True).item()
    # model = keras.models.load_model(path_model)

    # Grafico y guardo figuras
    img_folder = os.path.join('Informe', 'Figuras', '10_VGG16_cifar100')
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

    plt.plot(data['CAcc'], label="Acc. Training")
    plt.plot(data['val_CAcc'], label="Acc. Validation")
    # plt.title("Acc Test: {:.3f}".format(test_Acc))
    plt.xlabel("Epocas")
    plt.ylabel("Accuracy")
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(img_folder, 'Acc.pdf'),
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

    # graficos_6()

    # graficos_6_40N()

    graficos_6_CCorreccion()

    # graficos_7()

    # graficos_7_v2()

    # graficos_8_Densa()

    # graficos_8_Conv()

    # graficos_9_Densa()

    # graficos_9_Conv()

    # graficos_10_AlexNet_Cifar10()

    # graficos_10_AlexNet_Cifar100()

    # graficos_10_VGG_Cifar10()

    # graficos_10_VGG_Cifar100()
