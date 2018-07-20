from os.path import join, dirname
import pandas as pd
import numpy as np
from IPython.core.display import SVG
from keras.layers import Dense, Dropout, Activation
from keras.models import Sequential
from keras.utils.vis_utils import model_to_dot
from sklearn.preprocessing import LabelEncoder
import keras
import gensim

from models.unsupervised import text_normalizer as tn
from models.unsupervised import model_evaluation_utils as meu


def averaged_word2vec_vectorizer(corpus, model, num_features):
    vocabulary = set(model.wv.index2word)

    def average_word_vectors(words, model, vocabulary, num_features):
        features_vector = np.zeros((num_features,), dtype="float64")
        nwords = 0.
        for word in words:
            if word in vocabulary:
                nwords = nwords + 1
                features_vector = np.add(features_vector, model[word])
            if nwords:
                features_vector = np.divide(features_vector, nwords)
            return features_vector

    features = [average_word_vectors(tokenized_sentence, model, vocabulary, num_features)
                for tokenized_sentence in corpus]
    return np.array(features)


def construct_deepnn_architecture(num_input_features):
    dnn_model = Sequential()
    dnn_model.add(Dense(512, activation='relu', input_shape=(num_input_features,)))
    dnn_model.add(Dropout(0.2))
    dnn_model.add(Dense(512, activation='relu'))
    dnn_model.add(Dropout(0.2))
    dnn_model.add(Dense(512, activation='relu'))
    dnn_model.add(Dropout(0.2))
    dnn_model.add(Dense(2))
    dnn_model.add(Activation('softmax'))

    dnn_model.compile(loss='categorical_crossentropy', optimizer='adam',
                      metrics=['accuracy'])
    return dnn_model


if __name__ == '__main__':
    # Data
    data_path = join(dirname(dirname(dirname(__file__))), "data", "movie_reviews.csv")
    data = pd.read_csv(data_path)
    reviews = np.array(data["review"])
    sentiments = np.array(data["sentiment"])

    X_train, y_train = reviews[:35000], sentiments[:35000]
    X_test, y_test = reviews[35000:], sentiments[35000:]

    norm_train_reviews = tn.normalize_corpus(X_train)
    norm_test_reviews = tn.normalize_corpus(X_test)

    # Tokenizer reviews & encode labels
    le = LabelEncoder()
    num_classes = 2

    tokenized_train = [tn.tokenizer.tokenize(text) for text in norm_train_reviews]
    y_tr = le.fit_transform(y_train)
    y_train = keras.utils.to_categorical(y_tr, num_classes)

    tokenized_test = [tn.tokenizer.tokenize(text) for text in norm_test_reviews]
    y_ts = le.fit_transform(y_test)
    y_test = keras.utils.to_categorical(y_ts, num_classes)

    # Word2vec Model
    w2v_num_feartures = 500
    w2v_model = gensim.models.Word2Vec(tokenized_train, size=w2v_num_feartures, window=150, min_count=10, sample=1e-3)
    # generate averaged word vector features from word2vec model
    avg_wv_train_features = averaged_word2vec_vectorizer(corpus=tokenized_train,
                                                         model=w2v_model, num_features=500)
    avg_wv_test_features = averaged_word2vec_vectorizer(corpus=tokenized_test,
                                                        model=w2v_model, num_features=500)

    # Feature engineering with Glove model
    train_nlp = [tn.nlp(item) for item in norm_train_reviews]
    train_glove_features = np.array(item.vector for item in train_nlp)

    test_nlp = [tn.nlp(item) for item in norm_test_reviews]
    test_glove_features = np.array(item.vector for item in test_nlp)

    w2v_dnn = construct_deepnn_architecture(num_input_features=500)
    SVG(model_to_dot(w2v_dnn, show_shapes=True, show_layer_names=False,
                     rankdir="TB").create(prog='dot', format='svg'))

    # Training
    batch_size = 100
    w2v_dnn.fit(avg_wv_train_features, y_train, epochs=5, batch_size=batch_size,
                shuffle=True, validation_split=0.1, verbose=1)

    # Evaluation
    y_pred = w2v_dnn.predict_classes(avg_wv_test_features)
    prediction = le.inverse_transform(y_pred)
    meu.display_model_performance_metrics(true_labels=y_test,
                                          predicted_labels=prediction,
                                          classes=["positive", "negative"])
