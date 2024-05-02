import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Memuat dataset Iris
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Standarisasi fitur
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reduksi dimensi menggunakan PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Menerapkan Naive Bayes
model = GaussianNB()
model.fit(X_pca, y)

# Memprediksi hasil
y_pred = model.predict(X_pca)

# Visualisasi hasil
colors = ['navy', 'turquoise', 'darkorange']
target_names = iris.target_names

plt.figure()
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], color=color, alpha=.8, lw=2,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('Visualisasi Iris Dataset dengan Naive Bayes')
plt.show()

