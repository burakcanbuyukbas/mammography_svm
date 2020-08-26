import numpy as np
import cv2
from glob import glob
import fnmatch
from sklearn.model_selection import train_test_split
import random
from sklearn.decomposition import IncrementalPCA
import gc
import h5py
from tqdm import tqdm
import os
from utils import *


def get_PCA_data():
    print("Loading data...")
    X = np.load("X.npy", mmap_mode='r')
    #Y = np.load("Y.npy", mmap_mode='r')
    n = X.shape[0]  # how many rows we have in the dataset

    X = X.flatten().reshape(n, 7500)

    chunk_size = 50  # how many rows we feed to IPCA at a time, the divisor of n
    ipca = IncrementalPCA(n_components=50, batch_size=16)

    for i in range(0, n // chunk_size):
        ipca.partial_fit(X[i * chunk_size: (i + 1) * chunk_size])
    print("Fitted.")
    X = ipca.transform(X)
    print("Transformed.")
    np.save('X.npy', X)

    #fit but transform to images one by one and save them to folders!

def do_incremental_pca(batch=20, components=50, path="data/X.h5", target="data_batches"):
    h5 = h5py.File(path, 'r')
    data = h5['data']
    i = 0
    n = data.shape[0]  # total size
    batch_size = batch  # batch size
    ipca = IncrementalPCA(n_components=components, batch_size=batch)
    print("Fitting initialized...")
    for i in tqdm(range(0, n // batch_size)):
        ipca.partial_fit(data[i * batch_size: (i + 1) * batch_size])
    print("Trasformation initialized...")
    for i in tqdm(range(0, n // batch_size)):
        X_ipca = ipca.transform(data[i * batch_size: (i + 1) * batch_size])
        np.save(target + "/X" + str(i) + ".npy", X_ipca)

    #np.save('Y.npy', Y)

def do_incremental_pca_on_test(batch=20, components=50, path="data/X.h5", target="data_batches"):
    h5 = h5py.File(path, 'r')
    data = h5['data']
    i = 0
    n = data.shape[0]  # total size
    batch_size = batch  # batch size
    ipca = IncrementalPCA(n_components=components, batch_size=batch)
    print("Fitting initialized...")
    for i in tqdm(range(0, n // batch_size)):
        ipca.partial_fit(data[i * batch_size: (i + 1) * batch_size])
    del data
    testdata = np.load(data_folder + "/X_test.npy")
    testshape = testdata.shape
    n = testdata.shape[0]
    print(testshape)
    testdata = testdata.flatten().reshape(testshape[0], testshape[1] * testshape[2] * testshape[3])
    print("Test trasformation initialized...")
    for i in tqdm(range(0, n // batch_size)):
        X_ipca = ipca.transform(testdata[i * batch_size: (i + 1) * batch_size])
        np.save(target + "/X_test" + str(i) + ".npy", X_ipca)
    del testdata
    valdata = np.load(data_folder + "/X_val.npy")
    valshape = valdata.shape
    n = valdata.shape[0]
    print(valshape)
    valdata = valdata.flatten().reshape(valshape[0], valshape[1] * valshape[2] * valshape[3])
    print("Test trasformation initialized...")
    for i in tqdm(range(0, n // batch_size)):
        X_ipca = ipca.transform(valdata[i * batch_size: (i + 1) * batch_size])
        np.save(target + "/X_val" + str(i) + ".npy", X_ipca)

def concat_data(folder="data_batches", target="data/X_train.npy", count=0):
    arrays = []
    i = 0
    for i in tqdm(range(count)):
        X_add = np.load(folder + "/X_val" + str(i) + ".npy")
        arrays.append(X_add)
        print(X_add.shape)
    x = np.concatenate(arrays)
    np.save(target, x)
    print(x.shape)

def npy_to_npy_batches(size = 600):
    data = np.load(data_folder + "/X_train.npy")
    i = 0
    n = data.shape[0]  # total size
    batch_size = 600  # batch size

    for i in tqdm(range(0, n // batch_size)):
        xpart = (data[i * batch_size: (i + 1) * batch_size])
        xpart = xpart.flatten().reshape(batch_size, 128*128*3)
        np.save("data_batches/X" + str(i) + ".npy", xpart)


#npy_to_npy_batches()
#npy_to_h5(path='/X_train.npy')

#do_incremental_pca(batch=100, components=100, path="data/X_train.h5", target="data_batches/pca100")
#concat_data(folder="data_batches/pca100")

#do_incremental_pca_on_test(batch=100, components=100, path="data/X_train.h5", target="data_batches/pca100/test")
#concat_data(folder="data_batches/pca100/test", target="data/X_test", count=5)