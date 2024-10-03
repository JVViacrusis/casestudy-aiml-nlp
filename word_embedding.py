from pydoc import Helper

from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import helpers as Helper



class WordEmbedding:
    def __init__(self):
        # Download necessary NLTK data
        nltk.download('punkt')
        nltk.download('punkt_tab')
        nltk.download('stopwords')

        self.helper = Helper.DataSet("IMDB Dataset.csv", "sentiment")


        # Sample text corpus
        text_corpus = self.helper.get_X()["review"]
        # Tokenize the sentences
        tokenized_sentences = [word_tokenize(sentence.lower())
                               for sentence in text_corpus]

        # Remove stopwords (optional)
        stop_words = set(stopwords.words('english'))
        tokenized_sentences = [[word for word in sentence if word not in stop_words]
                               for sentence in tokenized_sentences]

        # Train the Word2Vec model on the tokenized sentences
        self.model = Word2Vec(sentences=tokenized_sentences,
                         vector_size=150, window=5, min_count=1, workers=4, seed=42)