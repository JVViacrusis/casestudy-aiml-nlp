import pandas as pd
from sentiment_analysis_cnn import SentimentAnalysisCNN, SentimentAnalysisCNNPreProcessor
from sklearn.metrics import accuracy_score


def get_sentences_and_labels(csv_path) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(csv_path)

    return df['review'].astype(str).tolist(), df['sentiment'].tolist()


def calculate_accuracy(actuals, predictions):
    accuracy = accuracy_score(actuals, predictions)

    # Calculate the percentage of matching elements
    percentage = accuracy * 100
    return percentage


def main():
    # Load data
    csv = 'IMDB Dataset.csv'
    sentences, sentiments = get_sentences_and_labels(csv)

    # Define train, test and valdation data sets
    test_size = 0.3
    validation_size = 0.3
    X_train, X_test, y_train, y_test = SentimentAnalysisCNNPreProcessor.split_data(
        sentences, sentiments, test_size)
    X_train, X_val, y_train, y_val = SentimentAnalysisCNNPreProcessor.split_data(
        X_train, y_train, validation_size)

    # Create word embeddings for the training and validation data
    X_train_embeddings, y_train_embeddings = SentimentAnalysisCNNPreProcessor.preprocess_data(
        X_train, y_train)
    X_val_embeddings, y_val_embeddings = SentimentAnalysisCNNPreProcessor.preprocess_data(
        X_val, y_val)

    # Train the model
    model = SentimentAnalysisCNN()
    model.fit(X_train_embeddings, y_train_embeddings,
              X_val_embeddings, y_val_embeddings)

    # Perform predictions with the model
    predictions = model.predict(X_test)

    # Calculate accuracy
    prediction_labels = [p[1] for p in predictions]
    accuracy = calculate_accuracy(y_test, prediction_labels)

    print(f"Model Accuracy: {accuracy}")


if __name__ == "__main__":
    main()
