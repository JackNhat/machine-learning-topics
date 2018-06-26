from os.path import join, dirname

import pandas as pd

import pydotplus
from sklearn import tree
from sklearn import preprocessing
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split


import matplotlib.pyplot as plt
import seaborn as sn

from data.preprocessing import rename_attributes


def fit_transform_ohe(df, col_name):
    """
    This function preform one hot frame encoding for the specified column.

    Args:
        df(pandas.DataFrame): the data frame containing the mentioned column name
        col_name: the couln to be hot encoder
    Returns:
        tuple: label_encoder, one_hot_encoder, transformed column as pandas Series
    """
    # label encode the column
    le = preprocessing.LabelEncoder()
    le_labels = le.fit_transform(df[col_name])
    df[col_name + "_label"] = le_labels

    # one hot encoding
    ohe = preprocessing.OneHotEncoder()
    feature_arr = ohe.fit_transform(df[[col_name + "_label"]]).toarray()
    feature_labels = [col_name + "_" + str(cls_label) for cls_label in le.classes_]
    features_df = pd.DataFrame(feature_arr, columns=feature_labels)
    return le, ohe, features_df


def transform_ohe(df, le, ohe, col_name):
    """This function performs one hot encoding for the specified
        column using the specified encoder objects.

    Args:
        df(pandas.DataFrame): the data frame containing the mentioned column name
        le(Label Encoder): the label encoder object used to fit label encoding
        ohe(One Hot Encoder): the onen hot encoder object used to fit one hot encoding
        col_name: the column to be one hot encoded

    Returns:
        tuple: transformed column as pandas Series

    """
    # label encode
    col_labels = le.transform(df[col_name])
    df[col_name + '_label'] = col_labels

    # ohe
    feature_arr = ohe.fit_transform(df[[col_name + '_label']]).toarray()
    feature_labels = [col_name + '_' + str(cls_label) for cls_label in le.classes_]
    features_df = pd.DataFrame(feature_arr, columns=feature_labels)

    return features_df


if __name__ == '__main__':
    data_path = join(dirname(dirname(dirname(__file__))), "data", "corpus", "hour.csv")
    hour_df = pd.read_csv(data_path)
    hour_df = rename_attributes(hour_df)
    X, X_test, y, y_test = train_test_split(hour_df.iloc[:, 0:-3],
                                                        hour_df.iloc[:, -1],
                                                        test_size=0.33,
                                                        random_state=42)

    X.reset_index(inplace=True)
    y_train = y.reset_index()

    X_test.reset_index(inplace=True)
    y_test = y_test.reset_index()

    cat_attr_list = ["season", "is_holiday", "weather_condition",
                     "is_workingday", "hour", "weekday", "month", "year"]
    numeric_feature_cols = ['temp', 'humidity', 'windspeed', 'hour', 'weekday', 'month', 'year']
    subset_cat_features = ['season', 'is_holiday', 'weather_condition', 'is_workingday']
    encoded_attr_list = []
    for col in cat_attr_list:
        return_obj = fit_transform_ohe(X, col)
        encoded_attr_list.append({"label_enc": return_obj[0],
                                  "ohe_enc": return_obj[1],
                                  "feature_df": return_obj[2],
                                  "col_name": col})
    feature_df_list = [X[numeric_feature_cols]]
    feature_df_list.extend([enc['feature_df'] for enc in encoded_attr_list if enc['col_name'] in subset_cat_features])
    train_df_new = pd.concat(feature_df_list, axis=1)

    # Training
    X = train_df_new
    y = y_train.total_count.values.reshape(-1, 1)
    dtr = DecisionTreeRegressor(max_depth=4, min_samples_split=5, max_leaf_nodes=10)
    dtr.fit(X, y)
    dot_data = tree.export_graphviz(dtr, out_file=None)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_pdf("bike_share.pdf")

    # Grid Search with Cross validation
    param_grid = {"criterion": ["mse", "mae"],
                  "min_samples_split": [10, 20, 40],
                  "max_depth": [2, 6, 8],
                  "min_samples_leaf": [20, 40, 100],
                  "max_leaf_nodes": [5, 20, 100, 500, 800]}
    grid_cv_dtr = GridSearchCV(dtr, param_grid, cv=5)
    grid_cv_dtr.fit(X, y)

    # Cross Validation: Best Model Details
    df = pd.DataFrame(data=grid_cv_dtr.cv_results_)
    fig, ax = plt.subplots()
    sn.pointplot(data=df[['mean_test_score',
                          'param_max_leaf_nodes',
                          'param_max_depth']],
                 y='mean_test_score', x='param_max_depth',
                 hue='param_max_leaf_nodes', ax=ax)
    ax.set(title="Effect of Depth and Leaf Nodes on Model Performance")
    fig.savefig("cross_validation_best_model.png")

    # Residual Plot
    predicted = grid_cv_dtr.best_estimator_.predict(X)
    residual = y.flatten() - predicted
    fig, ax = plt.subplots()
    ax.scatter(y.flatten(), residual)
    ax.axhline(lw=2, color='black')
    ax.set_xlabel("Observed")
    ax.set_ylabel("Residual")
    plt.savefig("resudual_plot.png")

    r2_scores = cross_val_score(grid_cv_dtr.best_estimator_, X, y, cv=10)
    mse_scores = cross_val_score(grid_cv_dtr.best_estimator_, X, y, cv=10, scoring='neg_mean_squared_error')
    best_dtr_model = grid_cv_dtr.best_estimator_

    # Test Dataset Performance
    test_encoded_attr_list = []
    for enc in encoded_attr_list:
        col_name = enc['col_name']
        le = enc['label_enc']
        ohe = enc['ohe_enc']
        test_encoded_attr_list.append({'feature_df': transform_ohe(X_test, le, ohe, col_name),
                                       'col_name': col_name})
    test_feature_df_list = [X_test[numeric_feature_cols]]
    test_feature_df_list.extend([enc['feature_df'] for enc in test_encoded_attr_list if enc['col_name'] in subset_cat_features])
    test_df_new = pd.concat(test_feature_df_list, axis=1)
    X_test = test_df_new
    y_test = y_test.total_count.values.reshape(-1, 1)
    y_predict = best_dtr_model.predict(X_test)
    residuals = y_test.flatten() - y_predict

    fig, ax = plt.subplots()
    ax.scatter(y_test.flatten(), residuals)
    ax.axhline(lw=2, color="black")
    ax.set_xlabel("Observed")
    ax.set_ylabel("Residual")
    plt.savefig("test_dataset_performance.png")
    r2_score = grid_cv_dtr.best_estimator_.score(X_test, y_test)
