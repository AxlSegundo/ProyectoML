# modulo_entrenamiento_rf.py

import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score, train_test_split

def entrenar_modelo_rf(X, y, test_size=0.2, random_state=42):
    # División de datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Inicializar y entrenar modelo
    clf = RandomForestClassifier(n_estimators=200, random_state=random_state)
    clf.fit(X_train, y_train)

    # Evaluación en test
    y_pred = clf.predict(X_test)

    print("\nReporte de Clasificación (Test):\n")
    print(classification_report(y_test, y_pred))

    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    plt.title("Matriz de Confusión - Random Forest")
    plt.show()

    # Validación cruzada
    cv_scores = cross_val_score(clf, X, y, cv=6)
    print(f"Accuracy promedio validación cruzada (5 folds): {cv_scores.mean():.4f}")

    return clf

if __name__ == "__main__":
    from Proyecto.prepro import preprocesar

    # Cargar y preprocesar datos
    X, y, le_drug, scaler = preprocesar("ProyectoML/Proyecto/drug200.csv")

    # Entrenar modelo y mostrar métricas
    modelo_rf = entrenar_modelo_rf(X, y)

    # Guardar modelo entrenado
    joblib.dump(modelo_rf, "ProyectoML/Proyecto/modelo_random_forest.pkl")
