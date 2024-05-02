from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Memuat dataset Iris
data = load_iris()
X = data.data
y = data.target

# Membuat model Decision Tree
model = DecisionTreeClassifier()
model.fit(X, y)

# Visualisasi Decision Tree
plt.figure(figsize=(20,10))
plot_tree(model, filled=True, feature_names=data.feature_names, class_names=data.target_names)
plt.title('Visualisasi Decision Tree untuk Dataset Iris')
plt.show()
