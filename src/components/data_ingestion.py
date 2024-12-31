import os
import sys
import nltk
from nltk.corpus import gutenberg

from src.exception import CustomException
from src.logger import logging

from src.configs.configurations import DataIngestionConfig

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            nltk.download('gutenberg')

            logging.info('Read entire hamlet text')

            # load the dataset
            data = gutenberg.raw('shakespeare-hamlet.txt')

            os.makedirs(os.path.dirname(self.ingestion_config.hamlet_text_file_path), exist_ok=True)

            # save to a file
            with open(self.ingestion_config.hamlet_text_file_path,'w') as file:
                file.write(data)

            logging.info("Ingestion of the data is completed")

            return(self.ingestion_config.hamlet_text_file_path)
        
        except Exception as e:
            raise CustomException(e,sys)
                
if __name__ == "__main__":
    obj=DataIngestion()
    hamlet_text_file_path = obj.initiate_data_ingestion()



