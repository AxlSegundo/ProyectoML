import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from Proyecto.prepro import preprocesar
def entrenar_modelo_mlp(X, y, test_size=0.2, random_state=42):
    # División estratificada
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Definimos y entrenamos el modelo
    clf = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=random_state)
    clf.fit(X_train, y_train)

    # Evaluación
    y_pred = clf.predict(X_test)
    print("\nReporte de Clasificación (MLP):\n")
    print(classification_report(y_test, y_pred))


    # Validación cruzada
    cv_scores = cross_val_score(clf, X, y, cv=5)
    print(f"Accuracy promedio validación cruzada (5 folds): {cv_scores.mean():.4f}")

    return clf, X_test, y_test

