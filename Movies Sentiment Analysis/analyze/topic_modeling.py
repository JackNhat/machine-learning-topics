from os.path import join, dirname
import numpy as np
import pandas as pd
import pyLDAvis
import pyLDAvis.sklearn
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

from models.unsupervised import text_normalizer as tn
import analyze.topic_model_utils as tmu

if __name__ == '__main__':
    # DATA
    data_path = join(dirname(dirname(__file__)), "data", "movie_reviews.csv")
    data = pd.read_csv(data_path)
    reviews = np.array(data["review"])
    sentiments = np.array(data["sentiment"])

    X_train, y_train = reviews[:35000], sentiments[:35000]
    X_test, y_test = reviews[35000:], sentiments[35000:]

    norm_train_reviews = tn.normalize_corpus(X_train)
    norm_test_reviews = tn.normalize_corpus(X_test)

    # TRANSFORMER
    norm_reviews = norm_train_reviews + norm_test_reviews
    positive_reviews = [review for review, sentiment in zip(norm_reviews, sentiments)
                        if sentiment == "positive"]
    ptvf = TfidfVectorizer(use_idf=True, min_df=0.05, max_df=0.95, ngram_range=(1, 1), sublinear_tf=True)
    ptvf_features = ptvf.fit_transform(positive_reviews)

    negative_reviews = [review for review, sentiment in zip(norm_reviews, sentiments)
                        if sentiment == "negative"]
    ntvf = TfidfVectorizer(use_idf=True, min_df=0.05, max_df=0.95, ngram_range=(1, 1), sublinear_tf=True)
    ntvf_features = ntvf.fit_transform(negative_reviews)

    # TOPIC MODEL
    pyLDAvis.enable_notebook()
    total_topics = 10

    # Positive Sentiments
    pos_nmf = NMF(n_components=total_topics, random_state=42, alpha=0.1, l1_ratio=0.2)
    pos_nmf.fit(ptvf_features)

    # extract features and component weights
    pos_feature_names = ptvf.get_feature_names()
    pos_weights = pos_nmf.components_
    # extract and display topics and their components
    pos_topics = tmu.get_topics_terms_weights(pos_weights, pos_feature_names)
    pos_topics = tmu.print_topics_udf(topics=pos_topics, total_topics=total_topics,
                                      num_terms=15, display_weights=False)
    pyLDAvis.sklearn.prepare(pos_nmf, ptvf_features, ptvf, R=15)

    # Negative Sentiments
    neg_nmf = NMF(n_components=total_topics, random_state=42, alpha=0.1, l1_ratio=0.2)
    neg_nmf.fit(ntvf_features)

    # extract features and component weights
    neg_feature_names = ntvf.get_feature_names()
    neg_weights = neg_nmf.components_
    # extract and display topics and their components
    neg_topics = tmu.get_topics_terms_weights(neg_weights, neg_feature_names)
    neg_topics = tmu.print_topics_udf(topics=neg_topics, total_topics=total_topics,
                                      num_terms=15, display_weights=False)
    pyLDAvis.sklearn.prepare(neg_nmf, ntvf_features, ntvf, R=15)

