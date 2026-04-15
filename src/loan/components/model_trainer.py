import os
import joblib
import mlflow
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from src.loan.components.data_transformation import DataTransformation
from src.loan.components.data_ingestion import DataIngestion

class ModelTrainer:
    def __init__(self):
        # Ensure this matches your Streamlit app's loading path exactly
        self.model_path = os.path.join("artifacts", "loan", "model.pkl")
        
        # Set local MLflow tracking directory for GitHub Actions
        os.makedirs("mlruns", exist_ok=True)
        mlflow.set_tracking_uri(f"file://{os.path.abspath('mlruns')}")

    def initiate_model_trainer(self, train_arr, test_arr):
        """
        Trains the XGBoost model, logs to MLflow, and saves the model artifact.
        """
        try:
            # 1. Split features and target + Force float type to prevent "Female" string errors
            X_train, y_train = train_arr[:, :-1].astype(float), train_arr[:, -1].astype(float)
            X_test, y_test = test_arr[:, :-1].astype(float), test_arr[:, -1].astype(float)

            # 2. Start MLFlow run
            with mlflow.start_run():
                
                # Define model parameters (Removed deprecated use_label_encoder)
                params = {
                    "n_estimators": 100,
                    "learning_rate": 0.1,
                    "max_depth": 5,
                    "eval_metric": "logloss"
                }

                # 3. Initialize and fit
                model = XGBClassifier(**params)
                model.fit(X_train, y_train)

                # 4. Evaluate
                preds = model.predict(X_test)
                acc = accuracy_score(y_test, preds)

                # 5. Log to MLflow
                mlflow.log_params(params)
                mlflow.log_metric("accuracy", acc)
                mlflow.xgboost.log_model(model, "model")

                # 6. Save model locally for Streamlit Cloud
                os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
                joblib.dump(model, self.model_path)

                print(f"--- Model Training Complete. Accuracy: {acc:.4f}")
                return acc

        except Exception as e:
            print(f"Error in model training: {e}")
            raise e

# --- Standalone Testing logic ---     
if __name__ == "__main__":
    # Initialize components
    ingestion = DataIngestion()
    train_p, test_p = ingestion.initiate_data_ingestion()
    
    transformation = DataTransformation()
    train_array, test_array, _ = transformation.initiate_data_transformation(train_p, test_p)

    # Train model
    trainer = ModelTrainer()
    trainer.initiate_model_trainer(train_array, test_array)
