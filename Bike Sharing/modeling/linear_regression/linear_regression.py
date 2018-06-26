from os.path import join, dirname
import matplotlib.pyplot as plt
import numpy as np

from sklearn import preprocessing, linear_model
import pandas as pd
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

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
    X_train, X_test, y_train, y_test = train_test_split(hour_df.iloc[:, 0:-3],
                                                        hour_df.iloc[:, -1],
                                                        test_size=0.33,
                                                        random_state=42)

    X_train.reset_index(inplace=True)
    y_train = y_train.reset_index()

    X_test.reset_index(inplace=True)
    y_test = y_test.reset_index()

    cat_attr_list = ["season", "is_holiday", "weather_condition",
                     "is_workingday", "hour", "weekday", "month", "year"]
    numeric_feature_cols = ['temp', 'humidity', 'windspeed', 'hour', 'weekday', 'month', 'year']
    subset_cat_features = ['season', 'is_holiday', 'weather_condition', 'is_workingday']
    encoded_attr_list = []
    for col in cat_attr_list:
        return_obj = fit_transform_ohe(X_train, col)
        encoded_attr_list.append({"label_enc": return_obj[0],
                                  "ohe_enc": return_obj[1],
                                  "feature_df": return_obj[2],
                                  "col_name": col})
    feature_df_list = [X_train[numeric_feature_cols]]
    feature_df_list.extend([enc['feature_df'] for enc in encoded_attr_list if enc['col_name'] in subset_cat_features])
    train_df_new = pd.concat(feature_df_list, axis=1)

    # Training
    X = train_df_new
    y = y_train.total_count.values.reshape(-1, 1)
    lin_reg = linear_model.LinearRegression()

    # Validation
    predict = cross_val_predict(lin_reg, X, y, cv=10)
    fig1, ax = plt.subplots()
    ax.scatter(y, y - predict)
    ax.axhline(lw=2, color='black')
    ax.set_xlabel("Observed")
    ax.set_ylabel("Residual")
    plt.savefig("linear_regression/cross_validation_predict.png")

    r2_scores = cross_val_score(lin_reg, X, y, cv=10)
    mse_scores = cross_val_score(lin_reg, X, y, cv=10, scoring="neg_mean_squared_error")

    fig2, ax = plt.subplots()
    ax.plot([i for i in range(len(r2_scores))], r2_scores, lw=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('R-Squared')
    ax.title.set_text("Cross Validation Scores, Avg:{}".format(np.average(r2_scores)))
    plt.savefig("linear_regression/cross_validation_score.png")

    lin_reg.fit(X, y)

    # Testing
    test_encoded_attr_list = []
    for enc in encoded_attr_list:
        col_name = enc['col_name']
        le = enc['label_enc']
        ohe = enc['ohe_enc']
        test_encoded_attr_list.append({'feature_df': transform_ohe(X_test, le, ohe, col_name),
                                       'col_name': col_name})
        test_feature_df_list = [X_test[numeric_feature_cols]]
        test_feature_df_list.extend(
            [enc['feature_df'] for enc in test_encoded_attr_list if enc['col_name'] in subset_cat_features])
        test_df_new = pd.concat(test_feature_df_list, axis=1)
    X_test = test_df_new
    y_test = y_test.total_count.values.reshape(-1, 1)
    y_pred = lin_reg.predict(X_test)
    resuduals = y_test - y_pred
    fig3, ax = plt.subplots()
    ax.scatter(y_test, resuduals)
    ax.axhline(lw=2, color='black')
    ax.set_xlabel("Observed")
    ax.set_ylabel("Residuals")
    ax.title.set_text("Rediduals Plot with R-Squared={}".format(np.average(r2_scores)))
    plt.savefig("linear_regression/testing_linear_regression.png")
