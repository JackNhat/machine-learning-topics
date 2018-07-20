from os.path import join, dirname
import numpy as np
import pandas as pd
from lime.lime_text import LimeTextExplainer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from models.unsupervised import text_normalizer as tn


def interprect_classification_model_prediction(doc_index, norm_corpus, corpus,
                                               predictions_labels, explainer_obj):
    print("Test document index: {index}\nActual sentiment: {actual}"
          "\nPredicted sentiment: {predicted}".format(index=doc_index,
                                                      actual=predictions_labels[doc_index],
                                                      predicted=pipeline.predict([norm_corpus[doc_index]])))
    print("\nReview: ", corpus[doc_index])
    print("\nModel Prediction Probabilities: ")
    for probs in zip(classes, pipeline.predict_proba([norm_corpus[doc_index]])[0]):
        print(probs)
    exp = explainer.explain_instance(norm_corpus[doc_index],
                                     pipeline.predict_proba, num_features=10,
                                     labels=[1])
    exp.show_in_notebook()


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
    transformer = CountVectorizer(binary=False, min_df=0.0, max_df=1.0, ngram_range=(1, 2))
    train_features = transformer.fit_transform(X_train)

    # TRAINING
    model = LogisticRegression()
    model.fit(train_features, X_train)

    # CLASSIFICATION PIPELINE
    pipeline = make_pipeline(transformer, model)

    # PREDICTION CLASSED
    classes = list(pipeline.classes_)
    print(pipeline.predict(['the lord of the rings is an excellent movie',
                            'i hated the recent movie on tv, it was so bad']))
    explainer = LimeTextExplainer()
    interprect_classification_model_prediction(doc_index=100, norm_corpus=norm_test_reviews,
                                               corpus=X_test, predictions_labels=y_test,
                                               explainer_obj=explainer)

    interprect_classification_model_prediction(doc_index=128, norm_corpus=norm_test_reviews,
                                               corpus=X_test, predictions_labels=y_test,
                                               explainer_obj=explainer)
