import os 
import sys
from src.insurance.components.data_ingestion import DataIngestion
from src.insurance.components.data_transformation import DataTransformation
from src.insurance.components.model_trainer import ModelTrainer

def run_training_pipeline():
    """ Orchestrates the full ML pipeline from ingestion to training.
    """
    try:
        # Create the top-level artifacts directory if it doesnt exist
        os.makedirs("artifacts", exist_ok = True)
        
        print("\n" + "="*30)
        print("STARTING INSURANCE TRAINING PIPELINE")
        print("="*30)

        #1. Data Ingestion
        ingestion = DataIngestion()
        train_path, test_path = ingestion.initiate_data_ingestion()
        
        #2. Data Transformation
        transformation = DataTransformation()
        train_arr, test_arr, _ = transformation.initiate_data_transformation(train_path, test_path)

        # 3. Model Training
        trainer = ModelTrainer()
        score = trainer.initiate_model_trainer(train_arr, test_arr)

        print("="*30)
        print(f'PIPELINE SUCCESS. Insurance Model MAE: {score:.4f}')
        print("="*30 + "\n")

        return score

    except Exception as e:
        print(f"PIPELINE FAILED: {e}")
        raise e

# Logic to run script standalone for testing
if __name__ == "__main__":
    run_training_pipeline()