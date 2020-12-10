from sklearn.svm import SVC
import numpy as np
import gc
import pickle
import h5py
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix


data_folder = "C:/Users/Burak/PycharmProjects/mammography/data/mass_calc"


#X_train = np.load("data/nopca/X_train.npy")
Y_train = np.load(data_folder + "/Y_train.npy")

clf = SGDClassifier(shuffle=True)

print("Step 1:")

x1 = np.load("data_batches/X0.npy", mmap_mode='r')
y1 = Y_train[0:600]
clf.partial_fit(x1, y1, classes=[0, 1])
del x1, y1
print("Step 2:")

x2 = np.load("data_batches/X1.npy", mmap_mode='r')
y2 = Y_train[600:1200]
clf.partial_fit(x2, y2, classes=[0, 1])
del x2, y2

print("Step 3:")

x3 = np.load("data_batches/X2.npy", mmap_mode='r')
y3 = Y_train[1200:1800]
clf.partial_fit(x3, y3, classes=[0, 1])
del x3, y3

print("Step 4:")

x4 = np.load("data_batches/X3.npy", mmap_mode='r')
y4 = Y_train[1800:2400]
clf.partial_fit(x4, y4, classes=[0, 1])
del x4, y4


# save the classifier
with open('model2.pkl', 'wb') as savedmodel:
    pickle.dump(clf, savedmodel)

x_test = np.load(data_folder + "/X_test.npy")
size = x_test.shape[0]
x_test = x_test.flatten().reshape(size, 128*128*3)
y_test = np.load(data_folder + "/Y_test.npy")
print(clf.score(x_test, y_test))


tn, fp, fn, tp = confusion_matrix(y_true=y_test, y_pred=clf.predict(x_test)).ravel()

print(f'training set: true negatives: {tn}')
print(f'training set: true positives: {tp}')
print(f'training set: false negatives: {fn}')
print(f'training set: false positives: {fp}')