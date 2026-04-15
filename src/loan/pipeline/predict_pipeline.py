import os
import joblib
import pandas as pd
from src.loan.components.data_transformation import DataTransformation
from src.loan.components.model_trainer import ModelTrainer

class PredictPipeline:
    def __init__(self):
        # Use absolute paths to ensure Streamlit Cloud finds the files
        base_dir = os.getcwd() 
        self.model_path = os.path.join(base_dir, "artifacts", "loan", "model.pkl")
        self.preprocessor_path = os.path.join(base_dir, "artifacts", "loan", "preprocessor.pkl")

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}. Did the CML Action run?")

        self.model = joblib.load(self.model_path)
        self.preprocessor = joblib.load(self.preprocessor_path)

    def predict(self, features):
        scaled = self.preprocessor.transform(features)
        return self.model.predict(scaled)

class CustomData:
    # ADD the arguments to match the training data schema
    def __init__(self, Gender, Married, Dependents, Education, Self_Employed, 
                 ApplicantIncome, CoapplicantIncome, LoanAmount, 
                 Loan_Amount_Term, Credit_History, Property_Area):
        self.data = {
            "Gender": [Gender],
            "Married": [Married],
            "Dependents": [Dependents], 
            "Education": [Education],
            "Self_Employed": [Self_Employed],
            "ApplicantIncome": [ApplicantIncome],
            "CoapplicantIncome": [CoapplicantIncome],
            "LoanAmount": [LoanAmount],
            "Loan_Amount_Term": [Loan_Amount_Term],
            "Credit_History": [Credit_History],
            "Property_Area": [Property_Area]
        }

    def get_as_df(self):
        return pd.DataFrame(self.data)
