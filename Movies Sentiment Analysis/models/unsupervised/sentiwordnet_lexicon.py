from os.path import dirname, join
import pandas as pd
import numpy as np
from data import preprocessing

from nltk.corpus import sentiwordnet as swn
import text_normalizer as tn
import models.unsupervised.model_evaluation_utils as meu


def analyze_sentiment_sentiwordnet_lexicon(review,
                                           verbose=False):
    # tokenize and POS tag text tokens
    tagged_text = [(token.text, token.tag_) for token in tn.nlp(review)]
    pos_score = neg_score = token_count = obj_score = 0
    # get wordnet synsets based on POS tags
    # get sentiment scores if synsets are found
    for word, tag in tagged_text:
        ss_set = None
        if 'NN' in tag and list(swn.senti_synsets(word, 'n')):
            ss_set = list(swn.senti_synsets(word, 'n'))[0]
        elif 'VB' in tag and list(swn.senti_synsets(word, 'v')):
            ss_set = list(swn.senti_synsets(word, 'v'))[0]
        elif 'JJ' in tag and list(swn.senti_synsets(word, 'a')):
            ss_set = list(swn.senti_synsets(word, 'a'))[0]
        elif 'RB' in tag and list(swn.senti_synsets(word, 'r')):
            ss_set = list(swn.senti_synsets(word, 'r'))[0]
        # if senti-synset is found
        if ss_set:
            # add scores for all found synsets
            pos_score += ss_set.pos_score()
            neg_score += ss_set.neg_score()
            obj_score += ss_set.obj_score()
            token_count += 1

    # aggregate final scores
    final_score = pos_score - neg_score
    norm_final_score = round(float(final_score) / token_count, 2)
    final_sentiment = 'positive' if norm_final_score >= 0 else 'negative'
    if verbose:
        norm_obj_score = round(float(obj_score) / token_count, 2)
        norm_pos_score = round(float(pos_score) / token_count, 2)
        norm_neg_score = round(float(neg_score) / token_count, 2)
        # to display results in a nice table
        sentiment_frame = pd.DataFrame([[final_sentiment, norm_obj_score, norm_pos_score,
                                         norm_neg_score, norm_final_score]],
                                       columns=pd.MultiIndex(levels=[['SENTIMENT STATS:'],
                                                                     ['Predicted Sentiment', 'Objectivity',
                                                                      'Positive', 'Negative', 'Overall']],
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

    # Experiments unsupervised SentiWordNet Lexicon
    awesome = list(swn.senti_synsets('awesome', 'a'))[0]
    print("Positive Polarity Score: ", awesome.pos_score())
    print("Negative Polarity Score: ", awesome.neg_score())
    print("Objective Score: ", awesome.obj_score())
    for review, sentiment in zip(test_reviews[sample_review_ids], test_sentiments[sample_review_ids]):
        print("REVIEW: ", review)
        print("Actual Sentiment: ", sentiment)
        pred = analyze_sentiment_sentiwordnet_lexicon(review, verbose=True)
        print("-" * 60)
    predicted_sentiments = [analyze_sentiment_sentiwordnet_lexicon(review, verbose=False)
                            for review in norm_test_reviews]
    meu.display_model_performance_metrics(true_labels=test_sentiments,
                                          predicted_labels=predicted_sentiments,
                                          classes=["positive", "negative"])
