import sys
import numpy as np
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
from src.configs.configurations import PredictPipelineConfig
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


class PredictPipeline:
    def __init__(self, is_classifier):
        self.predict_pipeline_config = PredictPipelineConfig()

    def predict_next_word(self, text):
        try:
            logging.info(f"Loading the tokenizer...")
            tokenizer = load_object(self.predict_pipeline_config.tokenizer_path)

            logging.info("Loading keras model...")
            model = load_model(self.predict_pipeline_config.model_path)

            token_list = tokenizer.texts_to_sequences([text])[0]

            max_sequence_len = model.input_shape[1] + 1  # Retrieve the max sequence length from the model input shape

            if len(token_list) >= max_sequence_len:
                token_list = token_list[-(max_sequence_len-1):]  # Ensure the sequence length matches max_sequence_len-1

            token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')

            predicted = model.predict(token_list, verbose=0)
            predicted_word_index = np.argmax(predicted, axis=1)

            logging.info("Ready to Predict.")

            for word, index in tokenizer.word_index.items():
                if index == predicted_word_index:
                    logging.info(f"Predicted Word: {word}")
                    return word
                

            logging.info(f"Nothing predicted!")
            return None
        
        except Exception as e:
            raise CustomException(e,sys)

