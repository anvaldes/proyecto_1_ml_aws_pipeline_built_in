import boto3
import sagemaker
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
from sagemaker.workflow.steps import ProcessingStep
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession

#----------------------------------------------------------------------

# Parametros generales
role = "arn:aws:iam::613602870396:role/SageMakerExecutionRole"
region = "us-east-1"
bucket = "proyecto-1-ml"
year = "2025"
month = "06"

# Sesiones de Sagemaker
boto_session = boto3.Session(region_name=region)
sagemaker_session = sagemaker.Session(boto_session=boto_session, default_bucket=bucket)
pipeline_session = PipelineSession()  # ← ✅ Aquí va la corrección

#----------------------------------------------------------------------

# 1. Preprocessing

#Rutas en S3
base_input_path_prepro = f"s3://{bucket}/datasets/{year}_{month}"
output_s3_uri_prepro = f"s3://{bucket}/preprocessing/{year}_{month}"

# ✅ Processor usando imagen built-in
preprocessing_processor = ScriptProcessor(
    image_uri="683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn:1.0-1-cpu-py3",
    command=['python3'],
    role=role,
    instance_count=1,
    instance_type='ml.t3.medium',
    sagemaker_session=pipeline_session
)

# ✅ Paso de preprocesamiento
step_preprocessing = ProcessingStep(
    name="PreprocessingStep",
    processor=preprocessing_processor,
    code="pipeline/preprocessing.py",
    inputs=[
        ProcessingInput(source=base_input_path_prepro, destination="/opt/ml/processing/input")
    ],
    outputs=[
        ProcessingOutput(source="/opt/ml/processing/output", destination=output_s3_uri_prepro)
    ]
)

#----------------------------------------------------------------------

# 2. Hyperparemeter Tuning Job

from sagemaker.sklearn.estimator import SKLearn
from sagemaker.tuner import (
    HyperparameterTuner,
    IntegerParameter,
    ContinuousParameter
)
from sagemaker.inputs import TrainingInput
from sagemaker.workflow.steps import TuningStep

# Config

entry_point = "pipeline/train_hpt.py"
base_job_name =  "hpt_sklearn-job"

source_dir = "."  # directorio donde está train.py y requirements.txt
output_path = f"s3://{bucket}/output"

# Hiperparámetros constantes
static_hyperparams = {
    "year": year,
    "month": month
}

# Rango de hiperparámetros
hyperparameter_ranges = {
    "n_estimators": IntegerParameter(2, 10),
    "max_depth": IntegerParameter(2, 10)
}

# Estimador usando imagen de sklearn
sklearn_estimator = SKLearn(
    entry_point = entry_point,
    role = role,
    instance_type = "ml.m5.large",
    instance_count = 1,
    framework_version = "1.2-1",  # versión oficial de sklearn en SageMaker
    py_version = "py3",
    hyperparameters = static_hyperparams,
    output_path = output_path,
    base_job_name = base_job_name,
    dependencies = ["requirements.txt"]
)

# Tuner
tuner = HyperparameterTuner(
    estimator = sklearn_estimator,
    objective_metric_name = "f1_score",
    objective_type = "Maximize",
    hyperparameter_ranges = hyperparameter_ranges,
    metric_definitions = [
        {"Name": "f1_score", "Regex": "f1_score: ([0-9\\.]+)"}
    ],
    max_jobs = 40,
    max_parallel_jobs = 4,
    base_tuning_job_name = "hpt-xgb",
)

step_tuning = TuningStep(
    name="TuningStep",
    tuner=tuner
)

#----------------------------------------------------------------------

# 3. Registration of the best model

from sagemaker.model import Model
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.step_collections import RegisterModel

custom_model_name = f"best-xgb-model-{year}-{month}"

best_model = Model(
    image_uri=sklearn_estimator.training_image_uri(),
    model_data=step_tuning.get_top_model_s3_uri(top_k = 1, s3_bucket = bucket, prefix = "output"),
    role=role,
    sagemaker_session=pipeline_session
)

step_register_model = ModelStep(
    name="RegisterBestModel",
    step_args=best_model.create(instance_type="ml.m5.large"),
    depends_on = [step_tuning]
    )

#----------------------------------------------------------------------

# 4. Evaluation

from sagemaker.workflow.steps import ProcessingStep
from sagemaker.workflow.properties import PropertyFile

# Procesador
evaluation_processor = ScriptProcessor(
    image_uri="683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-xgboost:1.5-1",
    command=["python3"],
    instance_type="ml.t3.medium",
    instance_count=1,
    role=role,
    sagemaker_session=pipeline_session,
)

# PropertyFile (opcional, útil si quieres exportar métricas)
evaluation_report = PropertyFile(
    name="EvaluationReport",
    output_name="evaluation",
    path="report.json"
)

# Paso de evaluación
step_evaluation = ProcessingStep(
    name="EvaluateBestModel",
    processor=evaluation_processor,
    inputs=[
        ProcessingInput(
            source=step_tuning.get_top_model_s3_uri(top_k=1, s3_bucket=bucket, prefix="output"),
            destination="/opt/ml/processing/model"
        ),
        ProcessingInput(
            source=f"s3://{bucket}/preprocessing/{year}_{month}",
            destination="/opt/ml/processing/input"
        )
    ],
    outputs=[
        ProcessingOutput(
            source="/opt/ml/processing/evaluation",
            destination=f"s3://{bucket}/evaluation/{year}_{month}/",
            output_name = "evaluation"
        )
    ],
    code="pipeline/evaluate.py",
    property_files=[evaluation_report]
)

#----------------------------------------------------------------------

# PIPELINE

step_evaluation.add_depends_on([step_register_model])
step_tuning.add_depends_on([step_preprocessing])

# 📈 Definir pipeline
pipeline = Pipeline(
    name=f"PreprocessingPipeline-{year}-{month}",
    steps=[step_preprocessing, step_tuning, step_register_model, step_evaluation],
    #steps = [step_tuning, step_register_model, step_evaluation], # COMENTAR
    sagemaker_session=pipeline_session
)

# 🚀 Ejecutar pipeline
if __name__ == "__main__":
    print("📦 Subiendo definición del pipeline...")
    pipeline.upsert(role_arn=role)

    print("🚀 Ejecutando pipeline...")
    execution = pipeline.start()
    execution.wait()
    print("✅ Pipeline ejecutado con éxito.")

#----------------------------------------------------------------------
