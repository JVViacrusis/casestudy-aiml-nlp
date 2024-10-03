"""trains then evaluates the performance of a model"""

from sentiment_embedding_trainer import SentimentEmbeddingTrainer
from sentiment_predictor import SentimentPredictor
from sentiment_test_runner import SentimentTestRunner

if __name__ == "__main__":
    # train_filepath = './datasets/preprocessed/IMDB Dataset.csv'
    # test_filepath = './datasets/preprocessed/IMDB Dataset.csv'

    # train_filepath = './datasets/preprocessed/stanford_rotten_tomatoes_dataset.csv'
    # test_filepath = './datasets/preprocessed/stanford_rotten_tomatoes_dataset.csv'

    train_filepath = './datasets/preprocessed/IMDB Dataset.csv'
    test_filepath = './datasets/preprocessed/stanford_rotten_tomatoes_dataset.csv'


    sentiment_trainer = SentimentEmbeddingTrainer(train_filepath)
    sentiment_trainer.run()
    
    # Initialize the SentimentPredictor with pre-trained embedding matrix and tokenizer
    predictor = SentimentPredictor(model_path='sentiment_model.h5', tokenizer_path='tokenizer.pkl')

    # Initialize the test runner with the predictor and the IMDB dataset path
    test_runner = SentimentTestRunner(predictor, test_filepath, batch_size=8192)

    # Run the test and print the accuracy
    accuracy = test_runner.run_test()
    print(f"Final accuracy after training on {train_filepath} and testing on {test_filepath}: {accuracy:.2f}%")
