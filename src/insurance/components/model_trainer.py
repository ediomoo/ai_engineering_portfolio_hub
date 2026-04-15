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

    def initiate_model_trainer(self, train_arr, test_arr):
        """
        Trains the XGBoost model, logs to MLflow, and saves the model artifact.
        """
        try:
            # 1. Split features and target from the concatenated arrays
            X_train = train_arr[:, :-1]
            X_test = test_arr[:, :-1]
            
            
            y_train = train_arr[:, -1]
            y_test = test_arr[:, -1]
    

            # 2. Start MLFlow run for Experiment Tracking
            with mlflow.start_run():
                params = {'colsample_bytree': 1.0,
                'learning_rate': 0.01,
                'max_depth': 3,
                'n_estimators': 500,
                'subsample': 0.7}

                # 3. Initialize and fit the XGBoost Regressor
                model = XGBRegressor(**params)
                model.fit(X_train, y_train)

                # 4. Evaluate Performance
                preds = model.predict(X_test)
                mae_score = mae(y_test, preds)
            
                # 5. Log Parameters, Metrics and Model to MLflow
                mlflow.log_params(params)
                mlflow.log_metric("score", mae_score)
                mlflow.xgboost.log_model(model, "model")

                # 6. Save model locally for Streamlit Cloud to load the model during app runtime
                os.makedirs(os.path.dirname(self.model_path), exist_ok = True)
                joblib.dump(model, self.model_path)

                print(f'--- Model Training Complete. MAE: {mae_score:.4f}')
                return mae_score

        except Exception as e:
            # Provide clear feedback for debugging on Streamlit Cloud
            print(f"Error occured during model training: {e}")
            raise e

# --- Standalone Testing logic ---
if __name__ == "__main__":
    # Initialise components
    ingestion =  DataIngestion()
    train_path, test_path = ingestion.initiate_data_ingestion()
    
    # Correctly unpack the return values
    transformation = DataTransformation()
    train_arr, test_arr, _ = transformation.initiate_data_transformation(train_path, test_path)

    # Train model
    trainer = ModelTrainer()
    trainer.initiate_model_trainer(train_arr, test_arr)
    