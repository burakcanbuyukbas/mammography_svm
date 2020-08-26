from sklearn.svm import SVC
import numpy as np
import gc
import pickle
import joblib

print("Loading model...")
#load model from pkl
with open('model1.pkl', 'rb') as fin:
  model = pickle.load(fin)



print("Model loaded. Loading data.")

X_test = np.load("data/X_test.npy")
Y_test = np.load("data/Y_test.npy")

size = X_test.shape[0]
X_test = X_test.flatten().reshape(size*7500, 1)