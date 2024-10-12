import numpy as np
import pickle
import re
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

class SentimentPredictor:
    def __init__(self, model_path, tokenizer_path, max_sequence_length=100):
        self.model = load_model(model_path)

        with open(tokenizer_path, 'rb') as f:
            self.tokenizer = pickle.load(f)
        self.max_sequence_length = max_sequence_length

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

    def preprocess_reviews(self, reviews):
        """
        Preprocesses a batch of reviews and converts them to padded sequences.

        Parameters:
        - reviews (list of str): List of review texts.

        Returns:
        - Padded sequences for the batch of reviews.
        """
        # Preprocess each review
        cleaned_reviews = [self.preprocess_text(review) for review in reviews]

        # Tokenize and pad the sequences
        sequences = self.tokenizer.texts_to_sequences(cleaned_reviews)
        padded_sequences = pad_sequences(sequences, maxlen=self.max_sequence_length, padding='post')

        return padded_sequences

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

    def predict_batch_sentiment(self, reviews):
        """
        Predicts the sentiment for a batch of reviews.

        Parameters:
        - reviews (list of str): The review texts.

        Returns:
        - List of predicted sentiment labels ('positive' or 'negative').
        """
        # Preprocess reviews
        padded_sequences = self.preprocess_reviews(reviews)

        # Make predictions
        predictions = self.model.predict(padded_sequences)

        # Convert predictions to labels
        predicted_labels = ['positive' if p > 0.5 else 'negative' for p in predictions]

        return predicted_labels

# Example usage
if __name__ == "__main__":
    # Load the sentiment predictor with the paths to the saved files
    predictor = SentimentPredictor(model_path='sentiment_model.h5', tokenizer_path='tokenizer.pkl')

    # Predict the sentiment of a new review
    review = "This movie was absolutely fantastic! I loved it."
    predicted_sentiment = predictor.predict_sentiment(review)

    print(f"The predicted sentiment is: {predicted_sentiment}")
