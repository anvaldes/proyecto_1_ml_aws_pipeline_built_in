import argparse
import pandas as pd
import xgboost as xgb
from sklearn.metrics import f1_score
import boto3
import os
import io
import joblib

print('👋 Iniciando entrenamiento...')

if __name__ == "__main__":

    # Argumentos
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=str, required=True)
    parser.add_argument("--month", type=str, required=True)
    parser.add_argument("--n_estimators", type=int, required=True)
    parser.add_argument("--max_depth", type=int, required=True)
    args = parser.parse_args()

    print("🧾 Argumentos:")
    print(args)

    #----------------------------------------------------------------------
    # ✅ Lectura del dataset desde S3

    print("📥 Leyendo datasets desde S3...")

    bucket = "proyecto-1-ml"
    prefix = f"preprocessing/{args.year}_{args.month}"

    s3 = boto3.client("s3")

    def read_csv_from_s3(bucket, key):
        print(f"🔸 Leyendo s3://{bucket}/{key}")
        obj = s3.get_object(Bucket=bucket, Key=key)
        return pd.read_csv(io.BytesIO(obj["Body"].read()))

    # Parametrizado por año y mes
    X_train_path = f"{prefix}/X_train.csv"
    X_train = read_csv_from_s3(bucket, X_train_path)
    y_train_path = f"{prefix}/y_train.csv"
    y_train = read_csv_from_s3(bucket, y_train_path)

    X_val_path = f"{prefix}/X_val.csv"
    X_val = read_csv_from_s3(bucket, X_val_path)
    y_val_path = f"{prefix}/y_val.csv"
    y_val = read_csv_from_s3(bucket, y_val_path)

    #----------------------------------------------------------------------

    # Entrenar modelo
    print("Entrenando modelo...")
    model = xgb.XGBClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=42
    )

    model = model.fit(X_train, y_train)

    # Train

    y_train_pred_prob = model.predict_proba(X_train)[:, 1]
    y_train_pred = (y_train_pred_prob >= 0.25)

    f1_train = f1_score(y_train, y_train_pred, average = 'macro')
    print('f1 train:', round(f1_train*100, 2))


    # Val

    y_val_pred_prob = model.predict_proba(X_val)[:, 1]
    y_val_pred = (y_val_pred_prob >= 0.25)

    f1 = f1_score(y_val, y_val_pred, average = 'macro')
    print('f1_score:', round(f1*100, 2))

    #----------------------------------------------------------------------

    print("💾 Guardando modelo en /opt/ml/model/model.joblib...")
    os.makedirs("/opt/ml/model", exist_ok=True)
    model.save_model("/opt/ml/model/model.json")
    print("✅ Modelo guardado correctamente.")

    print("✅ Versión de XGBoost:", xgb.__version__)

    #----------------------------------------------------------------------
    
    print("✅ Entrenamiento y evaluación completados.")