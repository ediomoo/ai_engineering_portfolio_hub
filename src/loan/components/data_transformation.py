import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from src.loan.components.data_ingestion import DataIngestion


class DataTransformation:
    def __init__(self):
        self.preprocessor_path = os.path.join("artifacts", "loan", "preprocessor.pkl")

    def initiate_data_transformation(self, train_path, test_path):
        """
        Applies preprocessing (imputing, scaling, encoding) to train and test data.
        """
        try:
            # 1. Load split datasets
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            # 2. Define Features
            target_column = 'Loan_Status'
            drop_columns = [target_column, "Loan_ID"]

            
            X_train = train_df.drop(columns = drop_columns, axis = 1)
            y_train = train_df[target_column].map({"Y": 1, "N": 0})


            X_test = test_df.drop(columns = drop_columns, axis = 1)
            y_test = test_df[target_column].map({"Y": 1, "N": 0})


            # 3. Identify numerical and categorical columns automatically
            num_cols = X_train.select_dtypes(exclude = "object").columns
            cat_cols = X_train.select_dtypes(include = "object").columns

            # 4. Numerical Pipeline        
            num_pipeline = Pipeline([("imputer", SimpleImputer(strategy = "median")),
        ("scaler", StandardScaler())])

            # 5. Categorical Pipeline
            cat_pipeline = Pipeline([("imputer", SimpleImputer(strategy = "most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown = "ignore", sparse_output = False)),
        ("scaler", StandardScaler(with_mean = False))])

            # 6. Combine pipeline into Preprocessor
            preprocessor = ColumnTransformer([("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols)])

        

            # 7. Execute transformations
            X_train_arr = preprocessor.fit_transform(X_train)
            X_test_arr = preprocessor.transform(X_test)

            train_arr = np.c_[X_train_arr, np.array(y_train)]
            test_arr = np.c_[X_test_arr, np.array(y_test)]

            # 8. Save the preprocessor object for PredictPipeline to use
            os.makedirs(os.path.dirname(self.preprocessor_path), exist_ok = True)
            joblib.dump(preprocessor, self.preprocessor_path)

            # Return concatenated arrays (Features + Target) and preprocessor path
            return train_arr, test_arr, self.preprocessor_path


        except Exception as e:
            # Provide clear feedback for debugging on Streamlit Cloud
            print(f"Error in data transformation: {e}")
            raise e

# --- Standalone Testing logic ---
if __name__ == '__main__':
    ingestion = DataIngestion()
    train_path, test_path = ingestion.initiate_data_ingestion()
    transformation = DataTransformation()
    transformation.initiate_data_transformation(train_path, test_path)

    
