import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

# Memuat dataset Iris
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Mengurangi dimensi dengan PCA untuk visualisasi
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

# Membuat model KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_reduced, y)

# Membuat grid untuk visualisasi
x_min, x_max = X_reduced[:, 0].min() - 1, X_reduced[:, 0].max() + 1
y_min, y_max = X_reduced[:, 1].min() - 1, X_reduced[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

# Memprediksi kelas untuk setiap titik di grid
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Visualisasi hasil
plt.contourf(xx, yy, Z, alpha=0.8)
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, edgecolors='k')
plt.title('Visualisasi K-Nearest Neighbors pada Data Iris')
plt.xlabel('Komponen Utama 1')
plt.ylabel('Komponen Utama 2')
plt.show()
