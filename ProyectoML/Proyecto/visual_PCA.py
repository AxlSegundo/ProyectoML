import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

def visualizar_pca(X, y, le_target):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    y_labels = le_target.inverse_transform(y)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y_labels, palette="Set2", s=60)
    plt.title("Visualización PCA (2 Componentes)")
    plt.xlabel("Componente Principal 1")
    plt.ylabel("Componente Principal 2")
    plt.legend(title="Medicamento")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
