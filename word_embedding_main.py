import numpy as np
from word_embedding import WordEmbeddingHelper, WordEmbeddingFactory
from helpers import DataSet

MODEL_FILE = "./word2vec.model" 


def get_average_vector(words, model):
    word_vectors = [model.wv[word] for word in words if word in model.wv]
    if not word_vectors:  # Handle case with no valid words
        return np.zeros(model.vector_size)
    return np.mean(word_vectors, axis=0)


def get_sentence_sentiment(sentence, model):
    words = sentence.split()  # Tokenize the sentence
    sentence_vector = get_average_vector(words, model)
    return sentence_vector


def cosine_similarity(A, B):
    dot_product = np.dot(A, B)  # Calculate dot product
    norm_A = np.linalg.norm(A)   # Calculate norm of A
    norm_B = np.linalg.norm(B)   # Calculate norm of B

    if norm_A == 0 or norm_B == 0:  # Avoid division by zero
        return 0.0
    return dot_product / (norm_A * norm_B)


def determine_sentiment(sentence, avg_positive, avg_negative, model):
    sentence_vector = get_sentence_sentiment(sentence, model)

    positive_similarity = cosine_similarity(sentence_vector, avg_positive)
    negative_similarity = cosine_similarity(sentence_vector, avg_negative)

    net_similarity = positive_similarity - negative_similarity

    if positive_similarity > negative_similarity:
        return "Positive", net_similarity
    elif negative_similarity > positive_similarity:
        return "Negative", net_similarity
    else:
        return "Neutral", net_similarity


def generate_model():
    dataset = DataSet("IMDB Dataset.csv")
    corpus = dataset.get_X().values.flatten().tolist()
    vector_dimensionality = 100

    model = WordEmbeddingFactory.generate_word_2_vec_model(corpus, vector_dimensionality)
    WordEmbeddingHelper.export_model_to_file(model, MODEL_FILE)


def get_model_accuracy():
    dataset = DataSet("IMDB Dataset.csv")
    X_train, X_test, y_train, y_test = dataset.get_train_test_split(
        train_size_percent=70, 
        test_size_percent=30
    )

    corpus = X_train.values.flatten().tolist()
    vector_dimensionality = 100
    # Load Word2Vec model (or generate it if not already done)
    model = WordEmbeddingFactory.generate_word_2_vec_model(corpus, vector_dimensionality)
    WordEmbeddingHelper.export_model_to_file(model, MODEL_FILE)  # Optionally save the model for later use

    # Define Positive and Negative Word Lists
    positive_words = ['good', 'great', 'fantastic', 'love', 'excellent', 'amazing']
    negative_words = ['bad', 'terrible', 'hate', 'awful', 'worst', 'horrible']

    # Calculate Average Vectors for Positive and Negative Words
    avg_positive_vector = get_average_vector(positive_words, model)
    avg_negative_vector = get_average_vector(negative_words, model)

    # Initialize a counter for correct predictions
    correct_predictions = 0

    # Loop through each sentence in the testing set
    for sentence, actual_sentiment in zip(X_test['review'], y_test):
        predicted_sentiment, _ = determine_sentiment(sentence, avg_positive_vector, avg_negative_vector, model)

        # Check if prediction matches actual label
        if predicted_sentiment.lower() == actual_sentiment.lower():
            correct_predictions += 1

    # Calculate accuracy
    accuracy = correct_predictions / len(y_test) * 100
    print(f"Model Accuracy: {accuracy:.2f}%")


def predict_using_model(input_sentence: str):
    # Load Word2Vec model
    file = MODEL_FILE
    model = WordEmbeddingHelper.load_model_from_file(file)

    # Define Positive and Negative Word Lists
    positive_words = ['good', 'great', 'fantastic',
                      'love', 'excellent', 'amazing']
    # positive_words = ['good']
    negative_words = ['bad', 'terrible', 'hate', 'awful', 'worst', 'horrible']
    # negative_words = ['bad']

    # Calculate Average Vectors for Positive and Negative Words
    avg_positive_vector = get_average_vector(positive_words, model)
    avg_negative_vector = get_average_vector(negative_words, model)

    # Analyzing Sentences
    sentiment = determine_sentiment(input_sentence, avg_positive_vector, avg_negative_vector, model)
    print(f"Sentence: '{input_sentence}' - Sentiment: {sentiment[0]} (Net Similarity: {sentiment[1]})")

def main():
    # get_model_accuracy()

    # "good bad" vs "bad good"
    sentence = "this is a fantastic movie"
    predict_using_model(sentence)

if __name__ == "__main__":
    main()
