import sys
import pandas as pd

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, load_object, evaluate_model
from src.configs.configurations import ModelTrainerConfig

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()


    def initiate_model_trainer(self, tokenizer_path, preprocessed_data_path):
        try:
            logging.info("Start to do train test split process.")

            # Read the CSV file
            df = pd.read_csv(preprocessed_data_path)

            # Convert the DataFrame to a numpy array
            input_sequences = df.values

            # Extract X and y
            X, y = input_sequences[:, :-1], input_sequences[:, -1]

            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            logging.info(f"Loading the tokenizer...")

            # Load the tokenizer object
            tokenizer = load_object(tokenizer_path)

            total_words=len(tokenizer.word_index)+1
            max_sequence_len = max([len(x) for x in input_sequences])

            logging.info(f"Info from the tokenizer:")
            logging.info(f"------------------------")
            logging.info(f"Total words (vocabulary size): {total_words}")
            logging.info(f"Max sequence length: {max_sequence_len}")
            logging.info(f"Shape of X: {X.shape}")
            logging.info(f"------------------------")

            # One-hot encode the labels
            y_train = to_categorical(y_train, num_classes=total_words)
            y_test = to_categorical(y_test, num_classes=total_words)

            logging.info(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
            logging.info(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

            logging.info("Start evaluating various parameters...")

            model_report:dict = evaluate_model(X_train = X_train, y_train = y_train,
                                                X_test = X_test, y_test = y_test, tokenizer = tokenizer)
            
            logging.info("Hyperparameter tuning completed.")

            best_model = model_report["best_model"]

            if best_model == None:
                raise CustomException("Training model failed")
            
            logging.info(f"Best model found.")
            logging.info(f"------------------------")
            logging.info(f"Best Params: {model_report['best_param']}")
            logging.info(f"Best Param Values: {model_report['best_param'].values()}")
            logging.info(f"------------------------")

            loss, accuracy = best_model.evaluate(X_test, y_test, verbose=1)

            logging.info(f"Test Loss: {loss}")
            logging.info(f"Test Accuracy: {accuracy}")

            # Save the trained model
            save_object(self.model_trainer_config.model_path, obj= best_model)

            logging.info("Best model saved.")

            logging.info(f"Final Accuracy: {accuracy}")
            logging.info(f"Complete Model Training")

            return accuracy           
        except Exception as e:
            raise CustomException(e,sys)
        

if __name__=="__main__":
    obj = DataIngestion()
    hamlet_text_file_path = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    tokenizer_path, preprocessed_data_path = data_transformation.initiate_data_transformation(hamlet_text_file_path)

    model_trainer = ModelTrainer()
    accuracy = model_trainer.initiate_model_trainer(tokenizer_path, preprocessed_data_path)
    print(f"Accuracy Score of the Trained Model is: {accuracy}")
    

