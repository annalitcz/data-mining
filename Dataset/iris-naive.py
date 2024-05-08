import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from matplotlib.gridspec import GridSpec
#done
# Memuat dataset Iris
data = load_iris()
X = data.data
y = data.target

# Membagi dataset menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) #96.67% 

# Membuat model Naive Bayes
model = GaussianNB()

# Melatih model dengan data latih
model.fit(X_train, y_train)

# Memprediksi hasil dengan data uji
y_pred = model.predict(X_test)

# Menghitung akurasi
accuracy = accuracy_score(y_test, y_pred)
print(f"Akurasi model: {accuracy * 100:.2f}%")

# Visualisasi hasil klasifikasi dengan PCA
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)  # Menggunakan data yang telah dinormalisasi

fig = plt.figure(figsize=(12, 6))
gs = GridSpec(1, 3, figure=fig)  # 1 baris, 3 kolom

# Subplot untuk PCA menggunakan 2 kolom
ax1 = fig.add_subplot(gs[0, :2])
scatter = ax1.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap='viridis', edgecolor='k', s=50)
ax1.set_title('Visualisasi Dataset Iris dengan PCA')
ax1.set_xlabel('Komponen Utama 1')
ax1.set_ylabel('Komponen Utama 2')
plt.colorbar(scatter, ax=ax1)

# Subplot untuk akurasi menggunakan 1 kolom
ax2 = fig.add_subplot(gs[0, 2])
ax2.bar(['Akurasi'], [accuracy * 100], color='blue')
ax2.set_ylabel('Persentase')
ax2.set_title('Akurasi Model Naive Bayes')
ax2.set_ylim([0, 100])
ax2.text(0, -10, f'{accuracy * 100:.2f}%', ha='center', va='top', color='black')

plt.tight_layout()
plt.show()