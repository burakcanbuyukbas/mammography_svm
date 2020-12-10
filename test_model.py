from sklearn.svm import SVC
import numpy as np
import gc
import pickle
import joblib
from utils import load_test

print("Loading model...")
# load model from pkl
# with open('model1.pkl', 'rb') as fin:
#   model = pickle.load(fin)

# load model from joblib
loaded_model = joblib.load("models/svmst.sav")


print("Model loaded. Loading data.")

X_test, Y_test = load_test()
# X_test = np.load("data/X_test.npy")
# Y_test = np.load("data/Y_test.npy")

size = X_test.shape
print(size)
X_test = X_test.flatten().reshape(size[0], size[1]*size[2]*size[3])

result = loaded_model.score(X_test, Y_test)
print(result)