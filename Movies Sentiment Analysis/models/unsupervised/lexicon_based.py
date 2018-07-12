from os.path import dirname, join
import pandas as pd
import numpy as np
from data import preprocessing

from afinn import Afinn

import models.unsupervised.model_evaluation_utils as meu

if __name__ == '__main__':
    # load and normalize data
    data_path = join(dirname(dirname(dirname(__file__))), "data", "movie_reviews.csv")
    dataset = pd.read_csv(data_path)
    reviews = np.array(dataset["review"])
    sentiments = np.array(dataset["sentiment"])
    test_reviews = reviews[20:]
    test_sentiments = reviews[20:]
    sample_review_ids = [7626, 3533, 13010]
    norm_test_reviews = preprocessing.normalize_corpus(test_reviews)

    # Experiments unsupervised Lexicon-Based models
    afn = Afinn(emoticons=True)

    for review, sentiment in zip(test_reviews[sample_review_ids], test_sentiments[sample_review_ids]):
        print("REVIEW: ", review)
        print("Actual Sentiment: ", sentiment)
        print("Predicted Sentiment polarity: ", afn.score(review))
        print("-"*60)
    sentiment_polarity = [afn.score(review) for review in reviews]
    predicted_sentiments = ['positive' if score >= 1.0 else 'negative'
                            for score in sentiment_polarity]
    meu.display_model_performance_metrics(true_labels=test_sentiments,
                                          predicted_labels=predicted_sentiments,
                                          classes=['positive', 'negative'])

