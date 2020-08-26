import h5py
import numpy as np
data_folder = "C:/Users/Burak/PycharmProjects/mammography/data/mass_calc"

testdata = np.load(data_folder + "/X_test.npy")
testshape = testdata.shape
print(testshape)