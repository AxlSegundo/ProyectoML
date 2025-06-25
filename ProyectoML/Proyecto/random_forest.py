from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score, train_test_split

def entrenar_modelo_rf(X, y, test_size=0.2, random_state=42):
    # División de datos
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

    # Validación cruzada
    cv_scores = cross_val_score(clf, X, y, cv=5)
    print(f"Accuracy promedio validación cruzada (5 folds): {cv_scores.mean():.4f}")

    return clf, X_test, y_test

