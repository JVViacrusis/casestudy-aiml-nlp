from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import numpy as np
import nltk
import ssl


class WordEmbeddingFactory:
    @staticmethod
    def generate_word2vec_embedding_matrix(corpus, word_vector_dimensionality, window=5, min_count=1, worker_threads=4):
        model = WordEmbeddingFactory.generate_word_2_vec_model(
            corpus, word_vector_dimensionality, window, min_count, worker_threads
        )
        return WordEmbeddingFactory.generate_embedding_matrix(model, word_vector_dimensionality)

    @staticmethod
    def generate_word_2_vec_model(corpus, word_vector_dimensionality, window=5, min_count=1, worker_threads=4) -> Word2Vec:
        tokenized_sentences = WordEmbeddingFactory._get_tokenized_sentences(
            corpus)
        model = Word2Vec(
            tokenized_sentences,
            vector_size=word_vector_dimensionality,
            window=window,
            min_count=min_count,
            workers=worker_threads
        )
        return model

    @staticmethod
    def generate_embedding_matrix(word_2_vec_model: Word2Vec, word_vector_dimensionality):
        # Vocabulary size
        vocabulary = word_2_vec_model.wv.index_to_key
        vocab_size = len(vocabulary)

        # Initialize embedding matrix with zeros, with dimensionality of (vocab_size x word_vector_dimensionality)
        embedding_matrix = np.zeros((vocab_size, word_vector_dimensionality))

        # Fill the embedding matrix with the word vectors
        for i, word in enumerate(vocabulary):
            embedding_matrix[i] = word_2_vec_model.wv[word]

        return embedding_matrix

    @staticmethod
    def _get_tokenized_sentences(corpus: list[str]):
        # Fix SSL certificate issue for nltk
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context

        # Download required nltk packages
        nltk.download('punkt_tab')
        nltk.download('punkt')
        nltk.download('stopwords')

        # Tokenize sentences
        tokenized_sentences = [word_tokenize(
            sentence.lower()) for sentence in corpus]

        # Remove stop words
        stop_words = set(stopwords.words('english'))
        tokenized_sentences = [
            [word for word in sentence if word not in stop_words]
            for sentence in tokenized_sentences
        ]

        return tokenized_sentences


class WordEmbeddingHelper:
    @staticmethod
    def export_model_to_file(model, file_path):
        model.save(file_path)

    @staticmethod
    def load_model_from_file(file_path) -> Word2Vec:
        print('model-file', file_path)
        return Word2Vec.load(file_path)
