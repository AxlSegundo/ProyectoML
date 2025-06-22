# analisis_final.py

from Proyecto.prepro import preprocesar
from Proyecto.random_forest import entrenar_modelo_rf
from Proyecto.neural_network import entrenar_modelo_mlp
from Proyecto.visual_PCA import visualizar_pca
from Proyecto.comparison import comparar_modelos, mostrar_matriz_confusion

def main():
    # Paso 1: Preprocesamiento
    X, y, le_drug, scaler = preprocesar("ProyectoML/Proyecto/drug200.csv")

    # Paso 2: Visualización PCA
    visualizar_pca(X, y, le_drug)

    # Paso 3: Entrenamiento de modelos
    modelo_rf, X_test_rf, y_test_rf = entrenar_modelo_rf(X, y)
    modelo_mlp, X_test_mlp, y_test_mlp = entrenar_modelo_mlp(X, y)

    # Paso 4: Comparación de desempeño con validación cruzada
    comparar_modelos(X, y, [modelo_rf, modelo_mlp], ["Random Forest", "MLP"])

    # Paso 5: Matrices de confusión individuales
    mostrar_matriz_confusion(modelo_rf, X_test_rf, y_test_rf, "Random Forest")
    mostrar_matriz_confusion(modelo_mlp, X_test_mlp, y_test_mlp, "MLP")

if __name__ == "__main__":
    main()
