import os
import joblib
import pandas as pd
from src.insurance.components.data_transformation import DataTransformation
from src.insurance.components.model_trainer import ModelTrainer

class PredictPipeline:
    def __init__(self):
        # Using absolute paths for Streamlit Cloud stability
        base_dir = os.getcwd()
        self.model_path = os.path.join(base_dir, "artifacts", "insurance", "model.pkl")
        self.preprocessor_path = os.path.join(base_dir, "artifacts", "insurance", "preprocessor.pkl")

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Insurance model not found at {self.model_path}")

        self.model = joblib.load(self.model_path)
        self.preprocessor = joblib.load(self.preprocessor_path)

    def predict(self, features):
        scaled = self.preprocessor.transform(features)
        # Returns the first element since it's a single prediction
        return self.model.predict(scaled)[0]

class CustomData:
    def __init__(self, Age, Sex, BMI, Children, Smoker, Region):
        # Ensure these keys match your CSV column names exactly (e.g., 'age' vs 'Age')
        self.data = {
            "age": [Age],
            "sex": [Sex],
            "bmi": [BMI],
            "children": [Children],
            "smoker": [Smoker],
            "region": [Region]
        }

    def get_as_df(self):
        return pd.DataFrame(self.data)
