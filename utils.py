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

data_folder = "C:/Users/Burak/PycharmProjects/mammography/data/mass_calc"

def save_data(X, Y, path="data"):
    print("Saving data...")
    np.save(path + '/X.npy', X)
    np.save(path + '/Y.npy', Y)

def save_data_train_test(X_train, Y_train, X_test, Y_test, path="data"):
    print("Saving data...")
    np.save(path + '/X_train.npy', X_train)
    np.save(path + '/Y_train.npy', Y_train)
    np.save(path + '/X_test.npy', X_test)
    np.save(path + '/Y_test.npy', Y_test)

def save_data_train_test_val(X_train, Y_train, X_test, Y_test, X_val, Y_val, path="data"):
    print("Saving data...")
    np.save(path + '/X_train.npy', X_train)
    np.save(path + '/Y_train.npy', Y_train)
    np.save(path + '/X_test.npy', X_test)
    np.save(path + '/Y_test.npy', Y_test)
    np.save(path + '/X_val.npy', X_val)
    np.save(path + '/Y_val.npy', Y_val)

def val_test_partition_data(X, Y, val_ratio, test_ratio):
    print("Partitioning data...")
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=val_ratio, random_state=42)
    X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=test_ratio, random_state=42)
    return X_train, X_val, X_test, Y_train, Y_val, Y_test

def test_partition_data(X, Y, test_ratio):
    print("Partitioning data...")
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_ratio, random_state=42)
    return X_train, X_test, Y_train, Y_test

def load_data(X_path='/X.npy', Y_path='/Y.npy'):
    print("Loading data...")
    X = np.load(data_folder + X_path, mmap_mode='r')
    Y = np.load(data_folder + Y_path, mmap_mode='r')

    return X, Y

def load_from_npy():
    print("Loading data...")
    X_Train = np.load(data_folder + '/X_train.npy', mmap_mode='r')
    X_Test = np.load(data_folder + '/X_test.npy', mmap_mode='r')
    Y_Train = np.load(data_folder + '/Y_train.npy', mmap_mode='r')
    Y_Test = np.load(data_folder + '/Y_test.npy', mmap_mode='r')
    print("Train Benign: " + str(np.count_nonzero(Y_Train == 0)))
    print("Train Malignant: " + str(np.count_nonzero(Y_Train == 1)))

    print("Test Benign: " + str(np.count_nonzero(Y_Test == 0)))
    print("Test Malignant: " + str(np.count_nonzero(Y_Test == 1)))

    return X_Train, Y_Train, X_Test, Y_Test

def load_train():
    print("Loading data...")
    X_Train = np.load(data_folder + '/X_train.npy', mmap_mode='r')
    Y_Train = np.load(data_folder + '/Y_train.npy', mmap_mode='r')
    print("Train Benign: " + str(np.count_nonzero(Y_Train == 0)))
    print("Train Malignant: " + str(np.count_nonzero(Y_Train == 1)))
    return X_Train, Y_Train

def load_test():
    print("Loading data...")
    X_Test = np.load(data_folder + '/X_test.npy', mmap_mode='r')
    Y_Test = np.load(data_folder + '/Y_test.npy', mmap_mode='r')

    print("Test Benign: " + str(np.count_nonzero(Y_Test == 0)))
    print("Test Malignant: " + str(np.count_nonzero(Y_Test == 1)))
    return X_Test, Y_Test

def npy_to_h5(path="/X_train.npy"):
    print("Converting npy to h5... For reasons.")
    X = np.load(data_folder + path, mmap_mode='r')
    size = X.shape[0]
    pixels = X.shape[1]
    data = X.flatten().reshape(size, pixels*pixels*3)

    with h5py.File('data/X_train.h5', 'w') as hf:
        hf.create_dataset("data", data=data)

