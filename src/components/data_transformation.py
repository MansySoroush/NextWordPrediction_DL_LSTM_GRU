import sys
import pandas as pd
import numpy as np
import dill

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords

from src.exception import CustomException
from src.logger import logging
from src.configs.configurations import DataTransformationConfig
from src.utils import save_object

from src.components.data_ingestion import DataIngestion


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        
    def initiate_data_transformation(self, hamlet_text_file_path):
        try:
            # load the dataset
            with open(hamlet_text_file_path,'r') as file:
                text = file.read().lower()

            logging.info("Reading entire hamlet text completed")
            logging.info("Tokenizing the text...")

            # Tokenize the text-creating indexes for words
            # out-of-vocabulary (OOV) token to handle rare or unseen words
            tokenizer = Tokenizer(oov_token="<OOV>")
            tokenizer.fit_on_texts([text])
            total_words = len(tokenizer.word_index)+1

            # Create input sequences
            input_sequences=[]
            for line in text.split('\n'):
                token_list = tokenizer.texts_to_sequences([line])[0]
                for i in range(1,len(token_list)):
                    n_gram_sequence=token_list[:i+1]
                    input_sequences.append(n_gram_sequence)

            # Finding the maximum length in sequences
            max_sequence_len = max([len(x) for x in input_sequences])

            # Pad Sequences
            input_sequences=np.array(pad_sequences(input_sequences,maxlen=max_sequence_len,padding='pre'))

            # Save the tokenizer
            save_object(
                file_path=self.data_transformation_config.tokenizer_path,
                obj=tokenizer,
                protocol= dill.HIGHEST_PROTOCOL
            )

            logging.info("Saved tokenizer")

            # Create a pandas DataFrame
            df = pd.DataFrame(input_sequences)

            # Save the DataFrame to a CSV file
            df.to_csv(self.data_transformation_config.preprocessed_data_path, index=False)

            logging.info("Saved processed data")
            logging.info("Complete preprocessing")

            return (
                self.data_transformation_config.tokenizer_path,
                self.data_transformation_config.preprocessed_data_path
            )

        except Exception as e:
            raise CustomException(e,sys)

if __name__=="__main__":
    obj = DataIngestion()
    hamlet_text_file_path = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    tokenizer_path, preprocessed_data_path = data_transformation.initiate_data_transformation(hamlet_text_file_path)
