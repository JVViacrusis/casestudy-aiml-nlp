import numpy as np
from keras.src.utils import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.python.keras.models import load_model

import word_embedding as word_embedding
from tensorflow.keras.preprocessing.text import Tokenizer

try:
    loaded_model = load_model('sentiment_model.h5')
except Exception as e:
    print(e)


# Create an LSTM model
model = Sequential()

embedding = word_embedding.WordEmbedding()

word2vec_model = embedding.model

train_X, test_X, train_y, test_y = embedding.helper.get_train_test_split(60, 40)

# Create a word to index dictionary
vocab_size = len(word2vec_model.wv.key_to_index)
embedding_dim = word2vec_model.vector_size
word_index = {word: index for index, word in enumerate(word2vec_model.wv.key_to_index)}

# Create an embedding matrix for the embedding layer of the LSTM
embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, idx in word_index.items():
    embedding_matrix[idx] = word2vec_model.wv[word]

# Tokenize the data and convert to sequences
tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.word_index = word_index
sequences = tokenizer.texts_to_sequences(train_X["review"])

labels = []
for sentiment in train_y:
    print(sentiment)
    if "positive" in sentiment:
        labels.append(1)
    elif "negative" in sentiment:
        labels.append(0)


#Test data
new_sentences = ["I love this product", "This is the worst"]
new_sequences = tokenizer.texts_to_sequences(new_sentences)
new_padded_sequences = pad_sequences(new_sequences, maxlen=10, padding='post')

X = pad_sequences(sequences, maxlen=10, padding='post')
y = np.array(labels)


# Add embedding layer, using pre-trained word embeddings from Word2Vec
model.add(Embedding(input_dim=vocab_size,
                    output_dim=embedding_dim,
                    weights=[embedding_matrix],
                    input_length=20,
                    trainable=False))  # Set trainable=False to keep the embeddings static

# Add LSTM layer
model.add(LSTM(128, return_sequences=False))

# Add Dense layer with sigmoid activation for binary sentiment output
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X, y, epochs=10, batch_size=2)
model.save("sentiment_model.h5")  # Saves as an HDF5 file


# Tokenize the test data and convert to sequences
tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.word_index = word_index
test_sequences = tokenizer.texts_to_sequences(test_X["review"])

# Pad the test sequences to match the length used during training
max_length = 100  # This should be the same as used in training
test_padded_sequences = pad_sequences(test_sequences, maxlen=max_length, padding='post')

# Convert y_test (sentiments) to a list of 0s and 1s
y_test = []
for sentiment in test_y:
    if "positive" in sentiment:
        y_test.append(1)
    elif "negative" in sentiment:
        y_test.append(0)

# Convert y_test to a numpy array for compatibility with Keras
y_test = np.array(y_test)

# Evaluate the model on the test data
loss, accuracy = model.evaluate(test_padded_sequences, y_test)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy * 100:.2f}%")



predictions = model.predict(new_padded_sequences)
# Threshold the predictions: Anything above 0.5 is class '1', otherwise class '0'
predicted_classes = (predictions > 0.5).astype("int32")

print(predicted_classes)
