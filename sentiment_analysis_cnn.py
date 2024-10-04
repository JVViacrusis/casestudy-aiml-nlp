# Import necessary libraries
import pandas as pd
import numpy as np
import re
import pickle

# Import Keras components
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Dense, Conv1D, GlobalMaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class SentimentAnalysisCNNPreProcessor:
    @staticmethod
    def preprocess_data(sentences: list[str], labels: list[str], max_vocab_size=10000, max_sequence_length=100):
        """
        Load and preprocess text and labels from the CSV file.

        Returns:
        - data: Padded sequences of tokenized texts.
        - labels: Binary-encoded sentiment labels.
        """
        data = SentimentAnalysisCNNPreProcessor.preprocess_sentences(
            sentences, max_vocab_size, max_sequence_length
        )
        labels = SentimentAnalysisCNNPreProcessor.preprocess_labels(labels)
        return data, labels

    @staticmethod
    def preprocess_sentences(sentences, max_vocab_size=10000, max_sequence_length=100):
        # Preprocess the text data
        texts = [
            SentimentAnalysisCNNPreProcessor._preprocess_text(text)
            for text in sentences
        ]

        # Tokenize the text data
        tokenizer = Tokenizer(num_words=max_vocab_size, oov_token='<OOV>')
        tokenizer.fit_on_texts(texts)
        sequences = tokenizer.texts_to_sequences(texts)

        print(f'Found {len(tokenizer.word_index)} unique tokens.')

        # Pad the sequences
        data = pad_sequences(
            sequences, maxlen=max_sequence_length, padding='post', truncating='post')

        return data

    @staticmethod
    def preprocess_labels(labels):
        # Convert labels to binary format (0 for negative, 1 for positive)
        label_encoder = LabelEncoder()
        labels = label_encoder.fit_transform(labels)

        return labels

    @staticmethod
    def split_data(data, labels, test_size) -> tuple:
        return train_test_split(data, labels, test_size=test_size, random_state=42)

    @staticmethod
    def _preprocess_text(text):
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


class SentimentAnalysisCNN:
    def __init__(self, embedding_dim=100, max_vocab_size=10000, max_sequence_length=100) -> None:
        self.model = None
        self.embedding_dim = embedding_dim
        self.max_vocab_size = max_vocab_size
        self.max_sequence_length = max_sequence_length

    def fit(self, X, y, X_val=None, y_val=None):
        self.model = self._build_model()

        early_stopping = EarlyStopping(monitor='val_loss', patience=2)
        self.model.fit(
            X,
            y,
            epochs=10,
            batch_size=32,
            validation_data=(X_val, y_val) if X_val.any(
            ) and y_val.any() else None,
            callbacks=[early_stopping]
        )

        return self

    def predict(self, sentences: list[str]) -> list[tuple[float, str]]:
        word_embeddings = SentimentAnalysisCNNPreProcessor.preprocess_sentences(
            sentences)

        return self.predict_word_embeddings(word_embeddings)

    def predict_word_embeddings(self, word_embeddings) -> list[tuple[float, str]]:
        prediction_values = self.model.predict(word_embeddings)

        def get_prediction_label(
            p_value): return 'positive' if p_value > 0.5 else 'negative'

        predictions = [
            (float(p_value), get_prediction_label(p_value)) for p_value in prediction_values
        ]

        return predictions

    def _build_model(self):
        model = Sequential()

        model.add(Embedding(input_dim=self.max_vocab_size,
                            output_dim=self.embedding_dim,
                            input_length=self.max_sequence_length))
        model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
        model.add(GlobalMaxPooling1D())
        # Sigmoid is a good mathematical activation function for binary classification
        model.add(Dense(1, activation='sigmoid'))

        model.compile(optimizer='adam',
                      loss='binary_crossentropy', metrics=['accuracy'])

        return model
