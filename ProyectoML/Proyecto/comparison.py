import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import cross_val_score
import joblib

def comparar_modelos(X, y, modelos, nombres):
    resultados = {}

    for modelo, nombre in zip(modelos, nombres):
        print(f"\nModelo: {nombre}")
        scores = cross_val_score(modelo, X, y, cv=5)
        print(f"Accuracy promedio (5-fold): {scores.mean():.4f}")
        resultados[nombre] = scores

    # Gráfica comparativa de boxplot
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=list(resultados.values()))
    plt.xticks(ticks=range(len(nombres)), labels=nombres)
    plt.ylabel("Accuracy")
    plt.title("Comparación de Validación Cruzada entre Modelos")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def mostrar_matriz_confusion(modelo, X_test, y_test, nombre_modelo):
    from sklearn.metrics import confusion_matrix

    y_pred = modelo.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Oranges")
    plt.title(f"Matriz de Confusión - {nombre_modelo}")
    plt.show()
