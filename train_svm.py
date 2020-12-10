from sklearn.svm import SVC
import numpy as np
import gc
import pickle
import h5py
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix
import joblib



#path = "C:/Users/Burak/PycharmProjects/mammography/data/mass_calc"
path = "data/lda"
# X_train = np.load(path + "/X_train.npy")
X_train = np.load(path + "/X_train.npy")
size = X_train.shape[0]

Y_train = np.load(path + "/Y_train.npy")[0:size]
print("X: " + str(X_train.shape))
print("Y: " + str(Y_train.shape))

# already flat
# X_train = X_train.flatten().reshape(size, 128*128*3)

svc = SVC(kernel='rbf', gamma='scale', verbose=True, max_iter=500000000)
#svc = SGDClassifier(learning_rate='constant', eta0=0.001, shuffle=True, verbose=True)
svc.fit(X_train, Y_train)

svc.score(X_train, Y_train)

del X_train, Y_train


X_test = np.load(path + "/X_val.npy")
testshape = X_test.shape
X_test = X_test.flatten().reshape(testshape[0], testshape[1])
Y_test = np.load(path + "/Y_val.npy")

# save the model to disk
filename = "models/lda.sav"
joblib.dump(svc, filename)

# load the model from disk
loaded_model = joblib.load(filename)
result = loaded_model.score(X_test, Y_test)
print(result)


tn, fp, fn, tp = confusion_matrix(y_true=Y_test, y_pred=svc.predict(X_test)).ravel()



#print(clf.score(x_test, y_test))
#tn, fp, fn, tp = confusion_matrix(y_true=y_test, y_pred=clf.predict(x_test)).ravel()

print(f'training set: true negatives: {tn}')
print(f'training set: true positives: {tp}')
print(f'training set: false negatives: {fn}')
print(f'training set: false positives: {fp}')