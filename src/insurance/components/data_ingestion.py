import os
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    """
     Configuration for data ingestion paths.
    Using type hints (field: type) is required for dataclasses.
    """

    # Placing artifacts specifically within the loan project sub-directory
    base_path: str = os.path.join("artifacts", "insurance")
    train_data_path: str = os.path.join(base_path, "train.csv")
    test_data_path: str = os.path.join(base_path, "test.csv")
    raw_data_path: str = os.path.join(base_path, "raw_data.csv")

class DataIngestion:
    """ Handles reading the raw dataset , splitting it into training/testing sets,
    and storing them as artifacts for the transformation and training stages.
    """
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        """ 
        Reads raw data, performs a train-test-split, and saves artifacts.
        """
        try:
            # Create the "artifacts/" directory if it doesn't exist
            os.makedirs(self.ingestion_config.base_path, exist_ok = True)

            # Load your raw dataset
            # Ensure "data/data.csv" is included in github repo!
            if not os.path.exists("data/insurance_data.csv"):
                raise FileNotFoundError("The Source file data/insurance_data.csv was not found.")

            df = pd.read_csv("data/insurance_data.csv")

            # Perform Train-Test Split
            train_set, test_set = train_test_split(df, test_size = .2, random_state = 42)

            # Save the raw data to the artifacts folder for record keeping
            df.to_csv(self.ingestion_config.raw_data_path, index = False)

             # Save the split datasets
            train_set.to_csv(self.ingestion_config.train_data_path, index = False)
            test_set.to_csv(self.ingestion_config.test_data_path, index = False)


            print("--- Data Ingestion and Train-Test split completed successfully")

            return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path

        except Exception as e:
            # Provide clear feedback for debugging on Streamlit Cloud
            print(f"Error occured during data ingestion {e}")
            raise e

# --- Standalone Testing logic ---
if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()

