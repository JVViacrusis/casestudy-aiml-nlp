import numpy as np
import pickle
import re
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences

class SentimentPredictor:
    def __init__(self, embedding_matrix_path, tokenizer_path, max_sequence_length=100):
        """
        Initializes the SentimentPredictor by loading the embedding matrix and tokenizer.

        Parameters:
        - embedding_matrix_path (str): Path to the saved embedding matrix.
        - tokenizer_path (str): Path to the saved tokenizer.
        - max_sequence_length (int): Maximum sequence length for padding input text.
        """
        self.embedding_matrix = np.load(embedding_matrix_path)
        self.max_sequence_length = max_sequence_length

        # Load tokenizer
        with open(tokenizer_path, 'rb') as f:
            self.tokenizer = pickle.load(f)

        # Build the prediction model
        self.model = self._build_model()

    def _build_model(self):
        """
        Builds the prediction model using the loaded embedding matrix.

        Returns:
        - model: Compiled Keras model for sentiment prediction.
        """
        vocab_size, embedding_dim = self.embedding_matrix.shape

        # Build the model architecture
        model = Sequential()
        model.add(Embedding(input_dim=vocab_size,
                            output_dim=embedding_dim,
                            input_length=self.max_sequence_length,
                            weights=[self.embedding_matrix],
                            trainable=False))  # Freeze the embedding weights
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        # Compile the model
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        return model

    @staticmethod
    def preprocess_text(text):
        """
        Preprocesses a review text by removing non-alphabetic characters and converting to lowercase.

        Parameters:
        - text (str): The raw review text.

        Returns:
        - Cleaned text.
        """
        text = re.sub(r"[^A-Za-z\s]", "", text)
        text = text.lower().strip()
        return text

    def predict_sentiment(self, review_text):
        """
        Predicts the sentiment of a given review text.

        Parameters:
        - review_text (str): The raw review text.

        Returns:
        - sentiment (str): 'positive' or 'negative'.
        """
        # Preprocess the review text
        cleaned_text = self.preprocess_text(review_text)

        # Tokenize the cleaned text
        sequence = self.tokenizer.texts_to_sequences([cleaned_text])

        # Pad the sequence to ensure it has the right length
        padded_sequence = pad_sequences(sequence, maxlen=self.max_sequence_length, padding='post')

        # Make a prediction
        prediction = self.model.predict(padded_sequence)

        # Convert the prediction into a sentiment label
        sentiment_value = prediction[0][0]
        sentiment_label = 'positive' if prediction[0][0] > 0.5 else 'negative'


        return sentiment_value, sentiment_label

# Example usage
if __name__ == "__main__":
    # Load the sentiment predictor with the paths to the saved files
    predictor = SentimentPredictor(embedding_matrix_path='embedding_matrix.npy', tokenizer_path='tokenizer.pkl')

    # Predict the sentiment of a new review
    review = "This movie was absolutely fantastic! I loved it."
    predicted_sentiment = predictor.predict_sentiment(review)

    print(f"The predicted sentiment is: {predicted_sentiment}")
