import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

try:
    from sklearn.cluster import KMeans  # check installation of sklearn
except:
    print("Not installed scikit-learn.")
    pass

data=pd.read_csv("data.csv")

plt.figure()
plt.grid()
plt.title("Sepal width and Sepal length")
plt.xlabel("Sepal width (cm)")
plt.ylabel("Sepal length (cm)")
plt.plot(data["Sepal width"], data["Sepal length"], linestyle='none', marker='o', color='purple', alpha=0.5)

plt.show()