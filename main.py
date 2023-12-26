import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
import numpy as np

# Объявление наборов данных
dataCSV = pd.read_csv("shopping_trends.csv")
file_size = os.path.getsize("shopping_trends.csv")
data_size = dataCSV.memory_usage(deep=True).sum()
number_of_rows = len(dataCSV)
categoricals_features = dataCSV.select_dtypes(include=['object']).columns.tolist()
numericals_features = dataCSV.select_dtypes(include=['int', 'float']).columns.tolist()
target_column = "Purchase Amount (USD)"
names_columns = dataCSV.columns

print(f'Размер файла: {file_size} байт')
print(f'Размер данных: {data_size} байт')
print(f'Количество строк в файле: {number_of_rows}')
print(f'Признаки и их типы данных: \n{dataCSV.dtypes}')


def search_passes(df):
    missing_values = df.isnull().sum()
    print(missing_values)


def numerical_graphics(df):
    for i in numericals_features:
        if i == 'ID':
            continue
        plt.figure(figsize=(12, 10))
        sns.kdeplot(df[f"{i}"], fill=True)
        plt.title(f'Гистограмма {i}')
        plt.show()


def categorical_graphics(df):
    for i in categoricals_features:
        plt.figure(figsize=(12, 10))
        if len(set(df[f"{i}"])) > 15:
            plt.xticks(rotation=90)
            plt.tick_params(axis='x', labelrotation=90)
        sns.histplot(df[f"{i}"], kde=False, color='skyblue')
        plt.xlabel(f"{i}")
        plt.ylabel('Частота распределения')
        plt.title(f'Гистограмма распределения {i}')
        plt.show()


def replacing_passes(df):
    for i in names_columns:
        if i == 'Gender' or i == 'ID':
            continue
        median_size = df.groupby('Gender')[f'{i}'].apply(lambda x: x.mode()[0]).reset_index()
        df = df.merge(median_size, on='Gender', how='left', suffixes=('', '_median'))
        df[f'{i}'].fillna(df[f'{i}_median'], inplace=True)
        df.drop(f'{i}_median', axis=1, inplace=True)
        df.to_csv('shopping_trends', index=False)


def encoding_of_categorical_values(df):
    label_encoder = LabelEncoder()
    for i in categoricals_features:
        df[f'{i}_encoded'] = label_encoder.fit_transform(df[f'{i}'])
        df.drop(f'{i}', axis=1, inplace=True)
        df.rename(columns={f'{i}': f'{i}'}, inplace=True)
    return df


def frequency_encoding(df):
    for i in categoricals_features:
        freq_encoding = df[f"{i}"].value_counts(normalize=True)
        df[f"{i}" + "_freq_enc"] = df[f"{i}"].map(freq_encoding)
        df.drop(f'{i}', axis=1, inplace=True)
    return df


def minmax_normalization(df):
    for i in numericals_features:
        df[f'{i}'] = (df[f'{i}'] - df[f'{i}'].min()) / (df[f'{i}'].max() - df[f'{i}'].min())
    return df


def z_normalization(df):
    for i in numericals_features:
        mean = df[f'{i}'].mean()
        std = df[f'{i}'].std()
        df[f'{i}_normalized'] = (df[f'{i}'] - mean) / std
    return df


def percentile(df):
    param = 'Age'
    lower_threshold = df[f'{param}'].quantile(0.01)
    upper_threshold = df[f'{param}'].quantile(0.01)
    filtered_df = df[(df[f'{param}'] >= lower_threshold) & (df[f'{param}'] <= upper_threshold)]


def remove_rare(df):
    i = "Gender"
    category_counts = df[f'{i}_encoded'].value_counts()
    threshold = len(df) / 100
    rare_categories = category_counts[category_counts < threshold].index.tolist()
    filtered_df = df[~df[f'{i}_encoded'].isin(rare_categories)]
    filtered_df.to_csv(df, index=False)


def save_csv(df, name):
    df.to_csv(name, index=False)


"""
search_passes(dataCSV)
numerical_graphics(dataCSV)
categorical_graphics(dataCSV)
replacing_passes(dataCSV)
search_passes(dataCSV)
"""

first_data = dataCSV.copy()
first_data = frequency_encoding(first_data)
second_data = dataCSV.copy()
second_data = encoding_of_categorical_values(second_data)


def heatmap(data):
    correlation_matrix = data.corr()
    plt.figure(figsize=(20, 13))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Тепловая карта корреляции')
    plt.show()


heatmap(first_data)
heatmap(second_data)
"""
first_data = minmax_normalization(first_data)
save_csv(first_data, "first_data")

second_data = dataCSV.copy()
second_data = encoding_of_categorical_values(second_data)
second_data = z_normalization(second_data)
save_csv(second_data, "second_data")

third_data = dataCSV.copy()
third_data = frequency_encoding(third_data)
third_data = z_normalization(third_data)
save_csv(third_data, "third_data")
"""

# Параметры
catboost_params = {
    "iterations": 100,
    "depth": 6,
    "learning_rate": 0.1
}
randomForest_params = {
    "n_estimators": 100,
    "max_depth": 5,
    "random_state": 42
}
gbm_params = {
    "n_estimators": 100,
    "learning_rate": 0.1,
    "max_depth": 5
}

adaBoost_params = {
    "n_estimators": 100,
    "learning_rate": 1.0,
    "loss": 'linear'
}


# Модель линейной регрессии
def linear_regression(data, target_column):
    data = data.drop('ID', axis=1)
    X = data.drop(columns=[target_column])
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print('MAE:', mae)
    print('MSE:', mse)
    print('R2:', r2)


# linear_regression(first_data, target_column)
# linear_regression(second_data, target_column)


def randomForest(data, target_column, randomForest_params={}):
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    model = RandomForestRegressor(**randomForest_params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print('MAE:', mae)
    print('MSE:', mse)
    print('R2:', r2)


randomForest(first_data, target_column, randomForest_params)
randomForest(second_data, target_column, randomForest_params)


# CatBoost
def catBoost(data, target_column, catboost_params={}):
    X = data.drop(columns=[target_column])
    y = data[target_column]
    model = CatBoostRegressor(**catboost_params)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("MAE:", mae)
    print("MSE:", mse)
    print("R2:", r2)


#catBoost(first_data, target_column, catboost_params)


def gradientBoosting(data, target_column, gbm_params):
    X = data.drop(columns=[target_column])
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = GradientBoostingRegressor()
    model.set_params(**gbm_params)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("MAE:", mae)
    print("MSE:", mse)
    print("R2:", r2)


#adaBoost
def adaBoostRegression(data, target_column, ada_params):
    X = data.drop(columns=[target_column])
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = AdaBoostRegressor(**ada_params)
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("MAE:", mae)
    print("MSE:", mse)
    print("R2:", r2)
