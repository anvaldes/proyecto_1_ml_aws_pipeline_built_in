import os
import pandas as pd
import joblib
import json
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
import xgboost as xgb
import tarfile

if __name__ == "__main__":

    print("Leyendo datasets")

    #----------------------------------------------------------------------

    # Train

    X_train_path = "/opt/ml/processing/input/X_train.csv"
    y_train_path = "/opt/ml/processing/input/y_train.csv"

    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path)

    #----------------------------------------------------------------------

    # Val

    X_val_path = "/opt/ml/processing/input/X_val.csv"
    y_val_path = "/opt/ml/processing/input/y_val.csv"

    X_val = pd.read_csv(X_val_path)
    y_val = pd.read_csv(y_val_path)

    #----------------------------------------------------------------------

    # Test

    X_test_path = "/opt/ml/processing/input/X_test.csv"
    y_test_path = "/opt/ml/processing/input/y_test.csv"

    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path)

    #----------------------------------------------------------------------

    print("✅ Versión de XGBoost:", xgb.__version__)

    print("Leyendo modelo")

    model_dir = "/opt/ml/processing/model"
    tar_path = os.path.join(model_dir, "model.tar.gz")

    # Extraer
    print(f"📦 Extrayendo {tar_path}")
    with tarfile.open(tar_path) as tar:
        tar.extractall(path=model_dir)
    
    print('-'*70)
    print("Archivos extraídos:")
    print(os.listdir(model_dir))
    print('-'*70)

    # Luego carga el modelo
    model_path = os.path.join(model_dir, "model.json")
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    
    print("Evaluation")

    #----------------------------------------------------------------------

    # Train

    y_train_pred_prob = model.predict_proba(X_train)[:, 1]
    y_train_pred = (y_train_pred_prob >= 0.25)

    f1_train = f1_score(y_train, y_train_pred, average = 'macro')
    print('f1 train:', round(f1_train*100, 2))

    #----------------------------------------------------------------------

    # Val

    y_val_pred_prob = model.predict_proba(X_val)[:, 1]
    y_val_pred = (y_val_pred_prob >= 0.25)

    f1_val = f1_score(y_val, y_val_pred, average = 'macro')
    print('f1 val:', round(f1_val*100, 2))

    #----------------------------------------------------------------------

    # Test

    y_test_pred_prob = model.predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_pred_prob >= 0.25)

    f1_test = f1_score(y_test, y_test_pred, average = 'macro')
    print('f1 test:', round(f1_test*100, 2))

    #----------------------------------------------------------------------

    report = classification_report(y_test, y_test_pred, output_dict=True)

    print("Guardar evaluation")

    os.makedirs("/opt/ml/processing/evaluation", exist_ok=True)
    with open("/opt/ml/processing/evaluation/report.json", "w") as f:
        json.dump(report, f)

    print("✅ Evaluación completada y guardada.")

    #----------------------------------------------------------------------