import pandas as pd
import os

print("🚀 Iniciando preprocesamiento...")

def process_file_X(filename):
    input_path = f"/opt/ml/processing/input/{filename}"
    output_path = f"/opt/ml/processing/output/{filename}"

    df = pd.read_csv(input_path)
    
    if "person_home_ownership" in df.columns:
        df["person_home_ownership"] = df["person_home_ownership"] * 2
    else:
        raise ValueError(f"La columna 'person_home_ownership' no se encuentra en {filename}")
    
    df.to_csv(output_path, index=False)
    print(f"✅ {filename} procesado y guardado en {output_path}")


def process_file_y(filename):
    input_path = f"/opt/ml/processing/input/{filename}"
    output_path = f"/opt/ml/processing/output/{filename}"
    
    df = pd.read_csv(input_path)
    
    df.to_csv(output_path, index=False)
    print(f"✅ {filename} procesado y guardado en {output_path}")


# Procesar los archivos
for name in ["X_train.csv", "X_val.csv", "X_test.csv"]:
    process_file_X(name)

for name in ["y_train.csv", "y_val.csv", "y_test.csv"]:
    process_file_y(name)
