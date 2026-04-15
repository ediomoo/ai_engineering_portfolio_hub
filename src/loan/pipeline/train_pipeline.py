import os
import sys
from src.loan.components.data_ingestion import DataIngestion
from src.loan.components.data_transformation import DataTransformation
from src.loan.components.model_trainer import ModelTrainer

def run_training_pipeline():
    """ Orchestrates the full ML pipeline from ingestion to training.
    """
    try:
        # Create the top-level artifacts directory if it doesnt exist
        os.makedirs("artifacts", exist_ok = True)

        print("\n" + "="*30)
        print("STARTING LOAN TRAINING PIPELINE")
        print("="*30)

        #1. Data Ingestion
        ingestion = DataIngestion()
        train_path, test_path = ingestion.initiate_data_ingestion()

        #2. Data Transformation
        transformation = DataTransformation()
        # We use "_" because we dont necessarily need the preprocessor path string here
        train_arr, test_arr, _ = transformation.initiate_data_transformation(train_path, test_path)
    
        # 3. Model Training
        trainer = ModelTrainer()
        score = trainer.initiate_model_trainer(train_arr, test_arr)
        
        print("="*30)
        print(f'PIPELINE SUCCESS. Loan Model Accuracy: {score:.4f}')
        print("="*30 + "\n")
        return score

    except Exception as e:
        print(f"PIPELINE FAILED: {e}")
        raise e

# Logic to run script standalone for testing
if __name__ == "__main__":
    run_training_pipeline()