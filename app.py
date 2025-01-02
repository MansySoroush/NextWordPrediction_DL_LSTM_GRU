import streamlit as st
import sys

from src.pipelines.predict_pipeline import PredictPipeline
from src.exception import CustomException
from src.logger import logging

# streamlit app
st.title("Next Word Prediction With LSTM-GRU")

if 'predicted' not in st.session_state:
    st.session_state['predicted'] = False

if not st.session_state['predicted']:
    st.session_state['predicted'] = True

    predict_pipeline = PredictPipeline(is_classifier=True)

    try:
        # User input
        input_text=st.text_input("Enter the sequence of Words", "To be or not to")
        if st.button("Predict Next Word"):
            logging.info(f"Data for Prediction: {input_text}")
            logging.info("Before Prediction")
            next_word = predict_pipeline.predict_next_word(input_text)
            st.write(f'Next word: {next_word}')

        st.session_state['predicted'] = False

    except Exception as e:
        raise CustomException(e,sys)

