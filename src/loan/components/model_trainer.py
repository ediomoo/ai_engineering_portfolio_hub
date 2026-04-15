import os
import joblib
import mlflow
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from src.loan.components.data_transformation import DataTransformation
from src.loan.components.data_ingestion import DataIngestion

class ModelTrainer:
    def __init__(self):
        self.model_path = os.path.join("artifacts", "loan", "model.pkl")

    def initiate_model_trainer(self, train_arr, test_arr):
        """
        Trains the XGBoost model, logs to MLflow, and saves the model artifact.
        """
        try:
            # 1. Split features and target from the concatenated arrays
            X_train, y_train =  train_arr[:, :-1], train_arr[:, -1]
            X_test, y_test = test_arr[:,:-1], test_arr[:, -1]
            

            # 2. Start MLFlow run for Experiment Tracking
            with mlflow.start_run():
                
                
                # Define model parameters
                params = {
                    "n_estimators": 100,
                    "learning_rate": 0.1,
                    "max_depth": 5,
                    "use_label_encoder": False,
                    "eval_metric": "logloss"

                }

                # 3. Initialize and fit the XGBoost Classifier
                model = XGBClassifier(**params)
                model.fit(X_train, y_train)

                # 4. Evaluate Performance
                preds = model.predict(X_test)
                acc = accuracy_score(y_test, preds)

                # 5. Log Parameters, Metrics and Model to MLflow
                mlflow.log_params(params)
                mlflow.log_metric("accuracy", acc)
                mlflow.xgboost.log_model(model, "model")

                # 6. Save model locally for Streamlit Cloud to load the model during app runtime
                os.makedirs(os.path.dirname(self.model_path), exist_ok = True)
                joblib.dump(model, self.model_path)

                print(f"Model Training Complete. Accuracy: {acc: 4f}")
                return acc

        except Exception as e:
            # Provide clear feedback for debugging on Streamlit Cloud
            print(f"Error in model training: {e}")
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