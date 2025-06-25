import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def cargar_datos(ruta_csv):
    df = pd.read_csv(ruta_csv)
    return df

def codificar_variables(df):
    le_sex = LabelEncoder()
    le_bp = LabelEncoder()
    le_chol = LabelEncoder()
    le_drug = LabelEncoder()

    df['Sex'] = le_sex.fit_transform(df['Sex'])
    df['BP'] = le_bp.fit_transform(df['BP'])
    df['Cholesterol'] = le_chol.fit_transform(df['Cholesterol'])
    df['Drug'] = le_drug.fit_transform(df['Drug'])

    return df, le_drug

def escalar_datos(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler

def preprocesar(ruta_csv):
    df = cargar_datos(ruta_csv)
    df, le_drug = codificar_variables(df)

    X = df.drop(columns=['Drug'])
    y = df['Drug']

    X_scaled, scaler = escalar_datos(X)

    return X_scaled, y, le_drug, scaler

