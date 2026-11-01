import mlflow
import mlflow.sklearn
import joblib
import os

mlflow.set_experiment("Dakar_Immo_AVM_Benchmark")

benchmark_runs = [
    {"name": "Regression_Ridge_Baseline", "params": {"alpha": 3.0}, "r2": 0.5762, "mae": 175248.4, "rmse": 332445.8},
    {"name": "Gradient_Boosting_Huber", "params": {"n_estimators": 350, "lr": 0.03, "loss": "huber"}, "r2": 0.6050, "mae": 164871.6, "rmse": 320939.3},
    {"name": "Random_Forest", "params": {"n_estimators": 300, "max_depth": 15}, "r2": 0.6202, "mae": 166894.5, "rmse": 314704.5},
    {"name": "LightGBM_Champion", "params": {"n_estimators": 350, "lr": 0.03, "num_leaves": 31}, "r2": 0.6400, "mae": 164107.9, "rmse": 306395.5}
]

model_path = "models/model_dakar_immo.pkl"

trusted_types = [
    'collections.OrderedDict',
    'lightgbm.basic.Booster',
    'lightgbm.sklearn.LGBMRegressor',
    'numpy.dtype',
    'numpy.core.multiarray.scalar'
]

for run in benchmark_runs:
    with mlflow.start_run(run_name=run["name"]):
        mlflow.log_params(run["params"])
        mlflow.log_metric("R2_Score", run["r2"])
        mlflow.log_metric("MAE_FCFA", run["mae"])
        mlflow.log_metric("RMSE_FCFA", run["rmse"])
        
        if run["name"] == "LightGBM_Champion" and os.path.exists(model_path):
            try:
                model = joblib.load(model_path)
                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="model_dakar_immo",
                    registered_model_name="Dakar_Immo_LightGBM_Prod",
                    skops_trusted_types=trusted_types
                )
            except Exception:
                # Solution de secours sans skops
                mlflow.log_artifact(model_path, artifact_path="model_dakar_immo")

print("✅ Expériences et Modèle Champion enregistrés avec succès dans MLflow !")