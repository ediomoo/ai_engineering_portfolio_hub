import os
import joblib 
import mlflow
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error as mae
from src.insurance.components.data_ingestion import DataIngestion
from src.insurance.components.data_transformation import DataTransformation

class ModelTrainer:
    def __init__(self):
        self.model_path = os.path.join("artifacts", "insurance", "model.pkl")
        
        # FIX: Standardize MLflow tracking for GitHub Actions/SQLite
        db_path = os.path.abspath("mlflow.db")
        mlflow.set_tracking_uri(f"sqlite:///{db_path}")
        mlflow.set_experiment("Insurance_Training")

    def initiate_model_trainer(self, train_arr, test_arr):
        """
        Trains the XGBoost model, logs to MLflow, and saves the model artifact.
        """
        try:
            # 1. Split features and target + Force float type
            X_train, y_train = train_arr[:, :-1].astype(float), train_arr[:, -1].astype(float)
            X_test, y_test = test_arr[:, :-1].astype(float), test_arr[:, -1].astype(float)

            # 2. Start MLFlow run
            with mlflow.start_run():
                params = {
                    'colsample_bytree': 1.0,
                    'learning_rate': 0.01,
                    'max_depth': 3,
                    'n_estimators': 500,
                    'subsample': 0.7
                }

                # 3. Initialize and fit the XGBoost Regressor
                model = XGBRegressor(**params)
                model.fit(X_train, y_train)

                # 4. Evaluate Performance
                preds = model.predict(X_test)
                mae_score = mae(y_test, preds)
            
                # 5. Log Parameters, Metrics and Model to MLflow
                mlflow.log_params(params)
                mlflow.log_metric("mae", mae_score)
                mlflow.xgboost.log_model(model, "model")

                # 6. Save model locally for Streamlit Cloud
                os.makedirs(os.path.dirname(self.model_path), exist_ok = True)
                joblib.dump(model, self.model_path)

                print(f'--- Model Training Complete. MAE: {mae_score:.4f}')
                return mae_score

        except Exception as e:
            print(f"Error occurred during model training: {e}")
            raise e

# --- Standalone Testing logic ---
if __name__ == "__main__":
    ingestion = DataIngestion()
    train_p, test_p = ingestion.initiate_data_ingestion()
    
    transformation = DataTransformation()
    train_array, test_array, _ = transformation.initiate_data_transformation(train_p, test_p)

    trainer = ModelTrainer()
    trainer.initiate_model_trainer(train_array, test_array)
