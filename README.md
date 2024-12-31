# Real World Deep Learning Project
## DL - RNN - LSTM - GRU
### Next Word Prediction

This project aims to develop a deep learning model for predicting the next word in a given sequence of words. 
The model is built using Long Short-Term Memory (LSTM-GRU) networks, which are well-suited for sequence prediction tasks. 
The project includes the following steps:

1- Data Collection: We use the text of Shakespeare's "Hamlet" as our dataset. This rich, complex text provides a good challenge for our model.

2- Data Preprocessing: The text data is tokenized, converted into sequences, and padded to ensure uniform input lengths. The sequences are then split into training and testing sets.

3- Model Building: An LSTM-GRU model is constructed with an embedding layer, two GRU & Dropout layers, and a dense output layer with a softmax activation function to predict the probability of the next word.

4- Model Training: The model is trained using the prepared sequences. In addition, BayesianOptimization is used to find the best parameters for the best model.

5- Model Evaluation: The model is evaluated using a set of example sentences to test its ability to predict the next word accurately.

6- Deployment: A Streamlit web application is developed to allow users to input a sequence of words and get the predicted next word in real-time.