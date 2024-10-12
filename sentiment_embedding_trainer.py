# Import necessary libraries
import pandas as pd
import numpy as np
import re
import os
import pickle

# Import Keras components
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

class SentimentEmbeddingTrainer:
    def __init__(self, csv_filepath, embedding_dim=100, max_vocab_size=10000, max_sequence_length=100):
        """
        Initializes the trainer with CSV file path and model parameters.

        Parameters:
        - csv_filepath (str): Path to the CSV file.
        - embedding_dim (int): Dimension of the embedding vectors.
        - max_vocab_size (int): Maximum number of words in the vocabulary.
        - max_sequence_length (int): Maximum length of input sequences.
        """
        self.csv_filepath = csv_filepath
        self.embedding_dim = embedding_dim
        self.max_vocab_size = max_vocab_size
        self.max_sequence_length = max_sequence_length
        self.tokenizer = None
        self.embedding_matrix = None

    def load_and_preprocess_data(self):
        """
        Load and preprocess text and labels from the CSV file.

        Returns:
        - data: Padded sequences of tokenized texts.
        - labels: Binary-encoded sentiment labels.
        """
        df = pd.read_csv(self.csv_filepath)

        # Ensure correct column names
        texts = df['review'].astype(str).tolist()   # The text of the reviews
        labels = df['sentiment'].tolist()           # The sentiment labels (positive/negative)

        # Preprocess the text data
        texts = [self.preprocess_text(text) for text in texts]

        # Convert labels to binary format (0 for negative, 1 for positive)
        label_encoder = LabelEncoder()
        labels = label_encoder.fit_transform(labels)

        # Tokenize the text data
        self.tokenizer = Tokenizer(num_words=self.max_vocab_size, oov_token='<OOV>')
        self.tokenizer.fit_on_texts(texts)
        sequences = self.tokenizer.texts_to_sequences(texts)

        print(f'Found {len(self.tokenizer.word_index)} unique tokens.')

        # Pad the sequences
        data = pad_sequences(sequences, maxlen=self.max_sequence_length, padding='post', truncating='post')

        return data, labels

    @staticmethod
    def preprocess_text(text):
        """
        Preprocess text by removing non-alphabetic characters and converting to lowercase.

        Parameters:
        - text (str): Raw text data.

        Returns:
        - Cleaned text.
        """
        text = re.sub(r"[^A-Za-z\s]", "", text)
        text = text.lower().strip()
        return text


    ########################################################## 
    #                                                        # 
    #                   just embedding                       # 
    #                  84.19% accuracy                       #
    #                                                        # 
    ########################################################## 
    # def build_model(self):
    #     """
    #     Build and compile a simple neural network model with an Embedding layer.

    #     Returns:
    #     - model: Compiled Keras model.
    #     """
    #     model = Sequential()
    #     model.add(Embedding(input_dim=self.max_vocab_size, output_dim=self.embedding_dim, input_length=self.max_sequence_length))
    #     model.add(Flatten())
    #     model.add(Dense(1, activation='sigmoid'))


    #     model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    #     return model

    ########################################################## 
    #                                                        # 
    #               embed plus conv1d+pooling                # 
    #                  87.46% accuracy                       #
    #                                                        # 
    ########################################################## 
    def build_model(self):
        from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D
        model = Sequential()
        model.add(Embedding(input_dim=self.max_vocab_size, 
                            output_dim=self.embedding_dim, 
                            input_length=self.max_sequence_length))
        model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
        model.add(GlobalMaxPooling1D())
        model.add(Dense(1, activation='sigmoid'))

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def train(self, X_train, y_train, X_val, y_val):
        """
        Train the model and store the trained embedding matrix.

        Parameters:
        - X_train: Training data.
        - y_train: Training labels.
        - X_val: Validation data.
        - y_val: Validation labels.

        Returns:
        - None
        """
        model = self.build_model()

        early_stopping = EarlyStopping(monitor='val_loss', patience=2)
        model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping])

        # Extract the trained embedding matrix
        self.embedding_matrix = model.layers[0].get_weights()[0]

        model.save('sentiment_model.h5')

    def save_embedding_and_tokenizer(self, embedding_output_path='embedding_matrix.npy', tokenizer_output_path='tokenizer.pkl'):
        """
        Save the trained embedding matrix and tokenizer to files.

        Parameters:
        - embedding_output_path (str): File path to save the embedding matrix.
        - tokenizer_output_path (str): File path to save the tokenizer.

        Returns:
        - None
        """
        np.save(embedding_output_path, self.embedding_matrix)
        print(f'Embedding matrix saved to {embedding_output_path}')
        print(f'Embedding matrix shape: {self.embedding_matrix.shape}')

        with open(tokenizer_output_path, 'wb') as f:
            pickle.dump(self.tokenizer, f)
        print(f'Tokenizer saved to {tokenizer_output_path}')

    def run(self, embedding_output_path='embedding_matrix.npy', tokenizer_output_path='tokenizer.pkl'):
        """
        Main method to load data, train the embedding model, and save the results.

        Parameters:
        - embedding_output_path (str): File path to save the embedding matrix.
        - tokenizer_output_path (str): File path to save the tokenizer.

        Returns:
        - None
        """
        # Load and preprocess data
        data, labels = self.load_and_preprocess_data()

        # Split the data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, random_state=42)

        # Train the model
        self.train(X_train, y_train, X_val, y_val)

        # Save the embedding matrix and tokenizer
        self.save_embedding_and_tokenizer(embedding_output_path, tokenizer_output_path)

# Example usage
if __name__ == "__main__":
    csv_filepath = 'IMDB Dataset.csv'  # Replace with your CSV file path
    sentiment_trainer = SentimentEmbeddingTrainer(csv_filepath)
    sentiment_trainer.run()
