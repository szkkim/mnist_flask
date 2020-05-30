import pickle
import pandas as pd
import numpy as np


test = pd.read_csv('test.csv')
test = test.iloc[0].astype('float32')
label = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
X = test / 255.0
X = X.values.reshape(-1, 28, 28, 1)

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

plt.xticks([])
plt.yticks([])
plt.grid(False)
plt.imshow(X[0].reshape(28, 28), cmap=plt.cm.binary)
plt.show()

with open( 'cnn_mnist.sav', 'rb') as f:
    model = pickle.load(f)
predicted_prob = model.predict_proba(X)
model_score = predicted_prob[:, 1]
recommendation = model.predict_classes(X)

print(model_score, recommendation)