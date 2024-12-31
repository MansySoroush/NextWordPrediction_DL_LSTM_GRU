import os
import sys
import dill
import numpy as np
import pandas as pd

from src.exception import CustomException
from src.logger import logging

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, Dropout, GRU
from tensorflow.keras.optimizers import Adam
from bayes_opt import BayesianOptimization
from bayes_opt import BayesianOptimization

def save_object(file_path, obj, protocol=dill.DEFAULT_PROTOCOL):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj, protocol)

    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def evaluate_model(X_train, y_train, X_test, y_test, tokenizer):
    try:
        def model_performance(embed_dim, gru_units_1, gru_units_2, dropout_1, dropout_2, learning_rate):
            # Build the model with given parameters
            model = Sequential()
            model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, 
                                output_dim=int(embed_dim), 
                                input_length=X_train.shape[1]))
            model.add(GRU(units=int(gru_units_1), return_sequences=True))
            model.add(Dropout(rate=dropout_1))
            model.add(GRU(units=int(gru_units_2)))
            model.add(Dropout(rate=dropout_2))
            model.add(Dense(units=len(tokenizer.word_index) + 1, activation='softmax'))
            model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
            
            # Train the model
            history = model.fit(X_train, y_train, epochs=5, batch_size=128, verbose=0, validation_data=(X_test, y_test))
            
            # Return validation accuracy as metric to optimize
            val_accuracy = history.history['val_accuracy'][-1]
            return val_accuracy

        # Define the parameter bounds
        pbounds = {
            'embed_dim': (50, 150),
            'gru_units_1': (50, 200),
            'gru_units_2': (50, 150),
            'dropout_1': (0.1, 0.5),
            'dropout_2': (0.1, 0.5),
            'learning_rate': (1e-4, 1e-2),
        }

        # Initialize the optimizer
        optimizer = BayesianOptimization(f=model_performance, pbounds=pbounds, random_state=42)
        optimizer.maximize(init_points=2, n_iter=5)  # Initial random trials + optimization iterations

        # Extract the best parameters
        best_params = optimizer.max['params']

        # Build the model with the best parameters
        best_model = Sequential()
        best_model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, 
                                output_dim=int(best_params['embed_dim']), 
                                input_length=X_train.shape[1]))
        best_model.add(GRU(units=int(best_params['gru_units_1']), return_sequences=True))
        best_model.add(Dropout(rate=best_params['dropout_1']))
        best_model.add(GRU(units=int(best_params['gru_units_2'])))
        best_model.add(Dropout(rate=best_params['dropout_2']))
        best_model.add(Dense(units=len(tokenizer.word_index) + 1, activation='softmax'))
        best_model.compile(optimizer=Adam(learning_rate=best_params['learning_rate']), loss='categorical_crossentropy', metrics=['accuracy'])
        
        # Train the best model on full training data
        best_model.fit(X_train, y_train, epochs=100, batch_size=128, validation_data=(X_test, y_test), verbose=1)
        
        # Return the report
        report = {
            "best_param": best_params,
            "best_model": best_model
        }
        return report

    except Exception as e:
        raise CustomException(e, sys)
