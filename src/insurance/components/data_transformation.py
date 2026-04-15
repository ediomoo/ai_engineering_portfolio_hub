import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from src.insurance.components.data_ingestion import DataIngestion


class DataTransformation:
    
    def __init__(self):
        self.preprocessor_path = os.path.join("artifacts", "insurance", "preprocessor.pkl")

    def initiate_data_transformation(self, train_path, test_path):
         """
        Applies preprocessing (imputing, scaling, encoding) to train and test data.
        """
        try:
            # 1. Load split datasets
            df_train = pd.read_csv(train_path)
            df_test = pd.read_csv(test_path)

            # 2. Define Features
            target_column = "charges"
            X_train = df_train.drop(columns = [target_column])
            X_test = df_test.drop(columns = [target_column])

            y_train = df_train[target_column]
            y_test = df_test[target_column]

            # 3. Identify numerical and categorical columns automatically
            num_cols = X_train.select_dtypes(exclude = "object").columns
            cat_columns = X_train.select_dtypes(include = "object").columns

            # 4. Numerical Pipeline 
            num_pipeline = Pipeline([("imputer", SimpleImputer(strategy = "median")),
            ("scaler", StandardScaler())])

            # 5. Categorical Pipeline
            cat_pipeline = Pipeline([("imputer", SimpleImputer(strategy = "most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown = "ignore", sparse_output = False)),
            ("scaler", StandardScaler(with_mean = False,))])

            # 6. Combine pipeline into Preprocessor
            preprocessor = ColumnTransformer([("num", num_pipeline, num_cols),
            ("cat", cat_pipeline, cat_cols)])

            # 7. Execute transformations
            X_train_arr = preprocessor.fit_transform(X_train)
            X_test_arr = preprocessor.transform(X_test)

            train_arr = np.c_[X_train, np.array(y_train)]
            test_arr = np.c_[X_test, np.array(y_test)]

            # 8. Save the preprocessor object for PredictPipeline to use
            os.makedirs(os.path.dirname(self.preprocessor_path), exist_ok = True)
            joblib.dump(preprocessor, self.preprocessor_path)

            # Return concatenated arrays (Features + Target) and preprocessor path
            return train_arr, test_arr, self.preprocessor_path

        except Exception as e:
            # Provide clear feedback for debugging on Streamlit Cloud
            print(f"Error occured during data transformation: {e}")
            raise e

# --- Standalone Testing logic ---
if __name__ == '__main__':
    obj = DataTransformation
    obj.initiate_data_transformation(DataIngestion().initiate_data_ingestion())

        