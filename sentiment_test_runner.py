import pandas as pd
from sklearn.metrics import accuracy_score
from concurrent.futures import ThreadPoolExecutor, as_completed
from sentiment_predictor import SentimentPredictor

class SentimentTestRunner:
    def __init__(self, predictor, csv_filepath, num_threads=8):
        """
        Initializes the SentimentTestRunner with the SentimentPredictor and dataset.

        Parameters:
        - predictor (SentimentPredictor): An instance of SentimentPredictor.
        - csv_filepath (str): Path to the CSV file containing the test data.
        - num_threads (int): Number of threads to use for multithreading.
        """
        self.predictor = predictor
        self.csv_filepath = csv_filepath
        self.num_threads = num_threads

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
        Runs the sentiment predictor on the test dataset, calculates accuracy at 1% intervals, and returns the final accuracy.

        Returns:
        - accuracy: The accuracy of the predictions as a percentage.
        """
        # Load the test data
        reviews, actual_labels = self.load_data()

        predicted_labels = []
        num_reviews = len(reviews)
        checkpoint_interval = num_reviews // 100  # Every 1% of dataset

        # Use ThreadPoolExecutor for multithreaded predictions
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            future_to_review = {executor.submit(self.predict_review, review): review for review in reviews}

            # Collect results as they complete
            for i, future in enumerate(as_completed(future_to_review), 1):
                predicted_labels.append(future.result())

                # Output accuracy every 1% of the dataset
                if i % checkpoint_interval == 0:
                    current_accuracy = accuracy_score(actual_labels[:i], predicted_labels) * 100
                    print(f"Processed {i/num_reviews*100:.2f}% of the dataset. Current accuracy: {current_accuracy:.2f}%")

        # Final accuracy on the entire dataset
        final_accuracy = accuracy_score(actual_labels, predicted_labels) * 100
        return final_accuracy

# Example usage
if __name__ == "__main__":
    # Assuming the SentimentPredictor class is already defined and imported
    # Initialize the SentimentPredictor with pre-trained embedding matrix and tokenizer
    predictor = SentimentPredictor(embedding_matrix_path='embedding_matrix.npy', tokenizer_path='tokenizer.pkl')

    # Initialize the test runner with the predictor and the IMDB dataset path
    test_runner = SentimentTestRunner(predictor, csv_filepath='IMDB Dataset.csv', num_threads=8)  # Adjust threads as needed

    # Run the test and print the accuracy
    accuracy = test_runner.run_test()
    print(f"Final accuracy of the model on the IMDB Dataset: {accuracy:.2f}%")
