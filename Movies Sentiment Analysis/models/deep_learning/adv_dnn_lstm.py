from collections import Counter
from os.path import join, dirname
import pandas as pd
import numpy as np
from keras import Sequential
from keras.layers import Embedding, SpatialDropout1D, LSTM, Dense
from keras_preprocessing import sequence
from sklearn.preprocessing import LabelEncoder

from models.unsupervised import text_normalizer as tn
from models.unsupervised import model_evaluation_utils as meu

if __name__ == '__main__':
    # DATA
    data_path = join(dirname(dirname(dirname(__file__))), "data", "movie_reviews.csv")
    data = pd.read_csv(data_path)
    reviews = np.array(data["review"])
    sentiments = np.array(data["sentiment"])

    X_train, y_train = reviews[:35000], sentiments[:35000]
    X_test, y_test = reviews[35000:], sentiments[35000:]

    norm_train_reviews = tn.normalize_corpus(X_train)
    norm_test_reviews = tn.normalize_corpus(X_test)

    tokenized_train = [tn.tokenizer.tokenize(text) for text in norm_train_reviews]
    tokenized_test = [tn.tokenizer.tokenize(text) for text in norm_test_reviews]

    # FEATURE ENGINEERING
    le = LabelEncoder()
    num_classes = 2
    max_len = np.max([len(review) for review in tokenized_train])

    token_counter = Counter([token for review in tokenized_train for token in review])
    vocab_map = {item[0]: index+1 for index, item in enumerate(dict(token_counter).items())}
    max_index = np.max(list(vocab_map.values()))
    vocab_map["PAD_INDEX"] = 0
    vocab_map["NOT_FOUND_INDEX"] = max_index + 1
    vocab_size = len(vocab_map)

    # Train reviews data corpus
    # Convert tokenized text reviews to numberic vectors
    train_X = [[vocab_map[token] for token in tokenized_review]
               for tokenized_review in tokenized_train]
    train_X = sequence.pad_sequences(train_X, maxlen=max_len)

    # Train prediction class labels
    # Convert text sentiments labels to binary encoding
    train_y = le.fit_transform(y_train)

    # Test reviews data corpus
    # Convert tokenized test reviews to numeric vectors
    test_X = [[vocab_map[token] if vocab_map.get(token) else vocab_map["NOT_FOUND_INDEX"]
               for token in tokenized_review]
              for tokenized_review in tokenized_test]
    test_X = sequence.pad_sequences(test_X, maxlen=max_len)
    # Test prediction class labels
    # Convert text sentiments labels to binary encoding
    test_y = le.fit_transform(y_test)

    # TRAINING MODEL
    EMBEDDING_DIM = 128
    LSTM_DIM = 64

    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM,
                        input_length=max_len))
    model.add(SpatialDropout1D(0.2))
    model.add(LSTM(LSTM_DIM, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(loss="binary_crossentropy", optimizer="adam",
                  metrics=["accuracy"])
    batch_size = 100
    model.fit(train_X, train_y, epochs=5, batch_size=batch_size,
              shuffle=True, validation_split=0.1, verbose=1)

    #  EVALUATE
    pred_test = model.predict_classes(test_X)
    predictions = le.inverse_transform(pred_test.flatten())
    meu.display_model_performance_metrics(true_labels=test_X,
                                          predicted_labels=predictions,
                                          classes=["positive", "negative"])
