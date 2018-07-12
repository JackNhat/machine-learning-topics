from os.path import dirname, join
import pandas as pd
import numpy as np
from nltk.sentiment import SentimentIntensityAnalyzer

from data import preprocessing

import models.unsupervised.model_evaluation_utils as meu
import models.unsupervised.text_normalizer as tn


def analyze_sentiment_vader_lexicon(review, threshold=0.1, verbose=False):
    review = tn.strip_html_tags(review)
    review = tn.remove_accented_chars(review)
    review = tn.expand_contractions(review)

    analyze = SentimentIntensityAnalyzer()
    scores = analyze.polarity_scores(review)
    agg_score = scores["compound"]
    final_sentiment = "positive" if agg_score >= threshold else "negative"

    if verbose:
        positive = str(round(scores['pos'], 2) * 100) + "%"
        final = round(agg_score, 2)
        negative = str(round(scores['neg'], 2) * 100) + "%"
        neutral = str(round(scores['neu'], 2) * 100) + "%"
        sentiment_frame = pd.DataFrame([[final_sentiment, final, positive, negative, neutral]],
                                       columns=pd.MultiIndex(levels=[["SENTIMENT STATS: "],
                                                                     ["Predicted Sentiment ", "Polarity Score",
                                                                      "Positive", "Negative", "Neutral"]],
                                                             labels=[[0, 0, 0, 0, 0], [0, 1, 2, 3, 4]]))
        print(sentiment_frame)
    return final_sentiment


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

    # Experiments unsupervised VADER Lexicon
    for review, sentiment in zip(test_reviews[sample_review_ids], test_sentiments[sample_review_ids]):
        print("REVIEW: ", review)
        print("Actual sentiment: ", sentiment)
        pred = analyze_sentiment_vader_lexicon(review, threshold=0.4, verbose=True)
        print('-'*60)
    predicted_sentiments = [analyze_sentiment_vader_lexicon(review, threshold=0.4, verbose=False)
                            for review in test_reviews]
    meu.display_model_performance_metrics(true_labels=test_sentiments,
                                          predicted_labels=predicted_sentiments,
                                          classes=['positive', 'negative'])
