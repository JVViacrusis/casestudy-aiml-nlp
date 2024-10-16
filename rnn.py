from pydoc import Helper

import numpy as np
from keras.src.layers import SimpleRNN
from keras.src.utils import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.python.keras.models import load_model

from tensorflow.keras.preprocessing.text import Tokenizer

from helpers import DataSet
from word_embedding import *


class RNNModel:

    def __init__(self):
        data = DataSet()

        # Create an LSTM model
        self.model = Sequential()

        self.max_length = 50

        #Create the model
        word2vec_model = WordEmbeddingFactory.generate_word_2_vec_model(data.get_X()["review"], 150)


        train_X, self.test_X, train_y, self.test_y = data.get_train_test_split()

        # Create a word to index dictionary
        vocab_size = len(word2vec_model.wv.key_to_index)
        embedding_dim = word2vec_model.vector_size
        word_index = {word: index for index, word in enumerate(word2vec_model.wv.key_to_index)}

        # Create an embedding matrix for the embedding layer of the LSTM
        embedding_matrix = np.zeros((vocab_size, embedding_dim))
        for word, idx in word_index.items():
            embedding_matrix[idx] = word2vec_model.wv[word]

        # Tokenize the data and convert to sequences
        self.tokenizer = Tokenizer(num_words=vocab_size)
        self.tokenizer.word_index = word_index
        sequences = self.tokenizer.texts_to_sequences(train_X["review"])

        #turn negatives and positives to numbers
        labels = []
        for sentiment in train_y:
            if "positive" in sentiment:
                labels.append(1)
            elif "negative" in sentiment:
                labels.append(0)


        X = pad_sequences(sequences, maxlen=self.max_length, padding='post')
        y = np.array(labels)


        # Add embedding layer, using pre-trained word embeddings from Word2Vec
        self.model.add(Embedding(input_dim=vocab_size,
                            output_dim=embedding_dim,
                            weights=[embedding_matrix],
                            trainable=False))  # Set trainable=False to keep the embeddings static

        # Add LSTM layer
        self.model.add(SimpleRNN(128, return_sequences=False))

        # Add Dense layer with sigmoid activation for binary sentiment output
        self.model.add(Dense(1, activation='sigmoid'))

        # Compile the model
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Train the model
        self.model.fit(X, y, epochs=10, batch_size=2)



    def accuracy(self):
        # Tokenize the test data and convert to sequences
        test_sequences = self.tokenizer.texts_to_sequences(self.test_X["review"])

        # Pad the test sequences to match the length used during training
        test_padded_sequences = pad_sequences(test_sequences, maxlen=self.max_length, padding='post')

        # Convert y_test (sentiments) to a list of 0s and 1s
        y_test = []
        for sentiment in self.test_y:
            if "positive" in sentiment:
                y_test.append(1)
            elif "negative" in sentiment:
                y_test.append(0)

        # Convert y_test to a numpy array for compatibility with Keras
        y_test = np.array(y_test)

        # Evaluate the model on the test data
        loss, accuracy = self.model.evaluate(test_padded_sequences, y_test)
        print(f"Test Loss: {loss}")
        print(f"Test Accuracy: {accuracy * 100:.2f}%")


    def test(self, sentence):
        # Test data
        new_sentences = sentence
        new_sequences = self.tokenizer.texts_to_sequences(new_sentences)
        new_padded_sequences = pad_sequences(new_sequences, maxlen=10, padding='post')
        predictions = self.model.predict(new_padded_sequences)
        # Threshold the predictions: Anything above 0.5 is class '1', otherwise class '0'
        predicted_classes = (predictions > 0.5).astype("int32")

        print([f"Predicted: {'Negative' if predict == 0 else 'Positive'}" for predict in predicted_classes])



rnn = RNNModel()
rnn.accuracy()