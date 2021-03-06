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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from numpy.random import *
from numpy.linalg import *


data_dir = "C:/Users/Burak/PycharmProjects/mammography/data/mass_calc/"

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

def apply_lda(components=1, path="data/X_train.h5", target="data/lda/"):
    X = np.load(data_dir + "X_train.npy", mmap_mode='r').astype('float32')
    X = X.flatten().reshape(2860, 128*128*3)

    # X, Y = get_data()
    Y = np.load(data_dir + "Y_train.npy")
    print(X.shape)
    lda = LinearDiscriminantAnalysis(solver='svd', n_components=components)
    lda.fit(X, Y)

    X_train = lda.transform(X)
    np.save(target + 'X_train.npy', X_train)
    np.save(target + 'Y_train.npy', Y)

    del X, Y

    X_val = np.load(data_dir + "X_val.npy", mmap_mode='r').astype('float32')
    Y_val = np.load(data_dir + "Y_val.npy")
    Xvalshape = X_val.shape
    X_val = X_val.flatten().reshape(Xvalshape[0], Xvalshape[1]*Xvalshape[2]*Xvalshape[3])

    X_val = lda.transform(X_val)
    np.save(target + 'X_val.npy', X_val)
    np.save(target + 'Y_val.npy', Y_val)

    del X_val, Y_val

    X_test = np.load(data_dir + "X_test.npy", mmap_mode='r').astype('float32')
    Y_test = np.load(data_dir + "Y_test.npy")
    Xtestshape = X_test.shape
    X_test = X_test.flatten().reshape(Xtestshape[0], Xtestshape[1]*Xtestshape[2]*Xtestshape[3])

    X_test = lda.transform(X_test)
    np.save(target + 'X_test.npy', X_test)
    np.save(target + 'Y_test.npy', Y_test)



apply_lda()

# npy_to_npy_batches()
# npy_to_h5(path='/X_train.npy')

# do_incremental_pca(batch=100, components=100, path="data/X_train.h5", target="data_batches/pca100")
# concat_data(folder="data_batches/pca100")

#do_incremental_pca_on_test(batch=100, components=100, path="data/X_train.h5", target="data_batches/pca100/test")
#concat_data(folder="data_batches/pca100/test", target="data/X_test", count=5)