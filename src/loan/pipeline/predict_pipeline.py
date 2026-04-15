import joblib
import pandas as pd

# Import custom logic for data paths and transformation steps
from src.loan.components.data_transformation import DataTransformation
from src.loan.components.model_trainer import ModelTrainer


class PredictPipeline:
    """ This class handles the end to end prediction process:
    Loading artifacts, transforming input, and generating results.
    """
    def __init__(self):
        # Initializing paths from the component classes
        self.model_path = ModelTrainer().model_path
        self.preprocessor_path = DataTransformation().preprocessor_path

        # Loading saved model and processor into memory
        self.model = joblib.load(self.model_path)
        self.preprocessor = joblib.load(self.preprocessor_path)

    def predict(self, features):
        """
         Takes raw input features and returns a prediction.
        """
        
        scaled = self.preprocessor.transform(features)
        return self.model.predict(scaled)

class CustomData:
    """
    Acts as a bridge between the User Interface and the Model.
    Maps Individual Input into a format the model understands.
    """

    # Mapping Constructor arguments to dictionary
    def __init__(self, Gender, Married, Education, Self_Employed, ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History, Property_Area):
        self.data = {
            "Gender": [Gender],
            "Married": [Married],
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
        """
        Converts dictionary into pandas dataframe.
        """
        return pd.DataFrame(self.data)
