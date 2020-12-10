from sklearn.svm import SVC
import numpy as np
import gc
import pickle
import h5py
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix
import joblib
from utils import load_from_npy


X_Train, Y_Train, X_Test, Y_Test = load_from_npy()

# already flat
X_Train = X_Train.flatten().reshape(X_Train.shape[0], X_Train.shape[1]*X_Train.shape[2]*X_Train.shape[3])

svc = SGDClassifier(shuffle=True)

#svc = SGDClassifier(learning_rate='constant', eta0=0.001, shuffle=True, verbose=True)
svc.fit(X_Train, Y_Train)

del X_Train, Y_Train


testshape = X_Test.shape
X_Test = X_Test.flatten().reshape(testshape[0], testshape[1]*testshape[2]*testshape[3])

# save the model to disk
filename = "models/sgdst.sav"
joblib.dump(svc, filename)

# load the model from disk
loaded_model = joblib.load(filename)
result = loaded_model.score(X_Test, Y_Test)
print(result)


tn, fp, fn, tp = confusion_matrix(y_true=Y_Test, y_pred=svc.predict(X_Test)).ravel()



#print(clf.score(x_test, y_test))
#tn, fp, fn, tp = confusion_matrix(y_true=y_test, y_pred=clf.predict(x_test)).ravel()

print(f'training set: true negatives: {tn}')
print(f'training set: true positives: {tp}')
print(f'training set: false negatives: {fn}')
print(f'training set: false positives: {fp}')


