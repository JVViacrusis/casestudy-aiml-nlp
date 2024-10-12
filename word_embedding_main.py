import numpy as np
from word_embedding import WordEmbeddingHelper
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

    # Use a threshold of 0.2 to determine the sentiment
    threshold = 0.2
    net_similarity = np.absolute(positive_similarity - negative_similarity)

    if positive_similarity - negative_similarity > threshold:
        return "Positive", net_similarity
    elif negative_similarity - positive_similarity > threshold:
        return "Negative", net_similarity
    else:
        return "Neutral", net_similarity

def main():
    # Load Word2Vec model
    file="./word2vec_model.model"
    model = WordEmbeddingHelper.load_model_from_file(file)

    # Define Positive and Negative Word Lists
    positive_words = ['good', 'great', 'fantastic', 'love', 'excellent', 'amazing']
    negative_words = ['bad', 'terrible', 'hate', 'awful', 'worst', 'horrible']

    # Calculate Average Vectors for Positive and Negative Words
    avg_positive_vector = get_average_vector(positive_words, model)
    avg_negative_vector = get_average_vector(negative_words, model)

    sentences = [
        "I love this movie! It was fantastic.",
        "The film was terrible and boring.",
        "It was an okay film, nothing special."
    ]

    # Analyzing Sentences
    for sentence in sentences:
        sentiment = determine_sentiment(sentence, avg_positive_vector, avg_negative_vector, model)
        print(f"Sentence: '{sentence}' - Sentiment: {sentiment[0]} (Net Similarity: {sentiment[1]})")


if __name__ == "__main__":
    main()