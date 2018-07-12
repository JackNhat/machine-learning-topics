from os.path import dirname, join
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from models.unsupervised import text_normalizer as tn
from models.unsupervised import model_evaluation_utils as meu
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

    # Transformer
    count_transformer = CountVectorizer(binary=False, min_df=0.0, max_df=1.0, ngram_range=(1, 2))
    tfidf_transformer = TfidfVectorizer(use_idf=True, min_df=0.0, max_df=1.0, ngram_range=(1, 2), sublinear_tf=True)
    count_train_features = count_transformer.fit_transform(norm_train_reviews)
    tfidf_train_features = tfidf_transformer.fit_transform(norm_train_reviews)
    count_test_features = count_transformer.transform(norm_test_reviews)
    tfidf_test_features = tfidf_transformer.transform(norm_test_reviews)

    # Train
    lr = LogisticRegression(penalty='l2', max_iter=100, C=1)
    svm = SGDClassifier(loss='hinge', n_iter=100)
    lr_bow_predictions = meu.train_predict_model(classifier=lr,
                                                 train_features=count_train_features,
                                                 train_labels=y_train,
                                                 test_features=count_test_features,
                                                 test_labels=y_test)
    meu.display_model_performance_metrics(true_labels=y_test,
                                          predicted_labels=lr_bow_predictions,
                                          classes=['positive', 'negative'])
