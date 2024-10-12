from word_embedding import WordEmbeddingFactory, WordEmbeddingHelper
from helpers import DataSet

def main():
    data = DataSet("IMDB Dataset.csv", "sentiment")
    word2vec_model = WordEmbeddingFactory.generate_word_2_vec_model(data.get_X()["review"], 100)
    WordEmbeddingHelper.export_model_to_file(word2vec_model, "word2vec_model.model")

    print(word2vec_model.wv.key_to_index)

if __name__ == "__main__":
    main()