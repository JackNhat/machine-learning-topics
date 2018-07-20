from os.path import join, dirname
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

if __name__ == '__main__':
    # DATA
    data_path = join(dirname(dirname(__file__)), "data", "movie_reviews.csv")
    data = pd.read_csv(data_path)
    reviews = np.array(data["review"])
    sentiments = np.array(data["sentiment"])

    X_train, y_train = reviews[:35000], sentiments[:35000]
    X_test, y_test = reviews[35000:], sentiments[35000:]

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
