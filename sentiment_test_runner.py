import pandas as pd
from sklearn.metrics import accuracy_score
from concurrent.futures import ThreadPoolExecutor, as_completed
from sentiment_predictor import SentimentPredictor

class SentimentTestRunner:
    def __init__(self, predictor, csv_filepath, batch_size):
        """
        Initializes the SentimentTestRunner with the SentimentPredictor and dataset.

        Parameters:
        - predictor (SentimentPredictor): An instance of SentimentPredictor.
        - csv_filepath (str): Path to the CSV file containing the test data.
        - num_threads (int): Number of threads to use for multithreading.
        """
        self.predictor = predictor
        self.csv_filepath = csv_filepath
        self.batch_size = batch_size

    def load_data(self):
        """
        Loads the review data and sentiment labels from the CSV file.

        Returns:
        - reviews: List of review texts.
        - actual_labels: List of actual sentiment labels ('positive' or 'negative').
        """
        df = pd.read_csv(self.csv_filepath)
        reviews = df['review'].astype(str).tolist()
        actual_labels = df['sentiment'].tolist()
        return reviews, actual_labels

    def predict_review(self, review):
        """
        Predicts the sentiment for a single review.

        Parameters:
        - review (str): The review text.

        Returns:
        - predicted_sentiment (str): Predicted sentiment ('positive' or 'negative').
        """
        return self.predictor.predict_sentiment(review)

    def run_test(self):
        """
        Runs the sentiment predictor on the test dataset and calculates accuracy in batches.

        Returns:
        - accuracy: The accuracy of the predictions as a percentage.
        """
        # Load the test data
        reviews, actual_labels = self.load_data()

        predicted_labels = []
        num_reviews = len(reviews)

        # Process in batches
        for start_idx in range(0, num_reviews, self.batch_size):
            end_idx = min(start_idx + self.batch_size, num_reviews)
            batch_reviews = reviews[start_idx:end_idx]

            # Predict sentiment for the batch
            batch_predicted_labels = self.predictor.predict_batch_sentiment(batch_reviews)
            predicted_labels.extend(batch_predicted_labels)

            # Output accuracy so far every 1% of the dataset
            if end_idx % (num_reviews // 100) == 0:
                current_accuracy = accuracy_score(actual_labels[:end_idx], predicted_labels) * 100
                print(f"Processed {end_idx/num_reviews*100:.2f}% of the dataset. Current accuracy: {current_accuracy:.2f}%")

        # Final accuracy on the entire dataset
        final_accuracy = accuracy_score(actual_labels, predicted_labels) * 100
        return final_accuracy

# Example usage
if __name__ == "__main__":
    # Initialize the SentimentPredictor with pre-trained embedding matrix and tokenizer
    predictor = SentimentPredictor(model_path='sentiment_model.h5', tokenizer_path='tokenizer.pkl')

    # Initialize the test runner with the predictor and the IMDB dataset path
    test_runner = SentimentTestRunner(predictor, csv_filepath='IMDB Dataset.csv', batch_size=8192)

    # Run the test and print the accuracy
    accuracy = test_runner.run_test()
    print(f"Final accuracy of the model on the IMDB Dataset: {accuracy:.2f}%")
