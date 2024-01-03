import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
import time
import csv
import random

# Объявление наборов данных
name_file = "ford.csv"
dataCSV = pd.read_csv(f"{name_file}")
file_size = os.path.getsize(f"{name_file}")
data_size = dataCSV.memory_usage(deep=True).sum()
number_of_rows = len(dataCSV)
categoricals_features = dataCSV.select_dtypes(include=['object']).columns.tolist()
numericals_features = dataCSV.select_dtypes(include=['int', 'float']).columns.tolist()
names_columns = dataCSV.columns
print(f"Наименование столбцов: {names_columns}")
print(f"Категориальные показатели: {categoricals_features}")
print(f"Количественные показатели: {numericals_features}")
target_column = "price"
scaler = StandardScaler()


def search_passes(df):
    missing_values = df.isnull().sum()
    print(missing_values)


def replacing_passes(df):
    for i in names_columns:
        if i == 'Id':
            continue
        median_size = df.groupby('model')[f'{i}'].apply(lambda x: x.mode()[0] if len(x) > 0 else None)
        median_size = median_size.reset_index(name=f'{i}_median')
        df = df.merge(median_size, on='model', how='left')
        df[f'{i}'].fillna(df[f'{i}_median'], inplace=True)
        df.drop(f'{i}_median', axis=1, inplace=True)
        df.to_csv(f"ford.csv", index=False)


def numerical_graphics(df):
    for i in numericals_features:
        if i == 'Id':
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


def encoding_of_categorical_values(df):
    label_encoder = LabelEncoder()
    for i in categoricals_features:
        df[f'{i}_encoded'] = label_encoder.fit_transform(df[f'{i}'])
        df.drop(f'{i}', axis=1, inplace=True)
        df.rename(columns={f'{i}_encoded': f'{i}'}, inplace=True)
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


def percentile(df):
    param = ''
    lower_threshold = df[f'{param}'].quantile(0.01)
    upper_threshold = df[f'{param}'].quantile(0.01)
    filtered_df = df[(df[f'{param}'] >= lower_threshold) & (df[f'{param}'] <= upper_threshold)]


def remove_rare(df):
    i = "model"
    category_counts = df[f'{i}'].value_counts()
    threshold = len(df) / 100
    rare_categories = category_counts[category_counts < threshold].index.tolist()
    filtered_df = df[~df[f'{i}'].isin(rare_categories)]
    return filtered_df


def smote(df, name_column):
    X = df.drop(name_column, axis=1)
    y = df[name_column]
    smote = SMOTE()
    X_resampled, y_resampled = smote.fit_resample(X, y)
    resampled_data = pd.concat([X_resampled, y_resampled], axis=1)
    return resampled_data


def balance_csv_dataset(df, name_column):
    class_counts = df[name_column].value_counts()
    max_count = class_counts.max()
    for class_label, count in class_counts.items():
        if count > max_count:
            remove_indices = df[df[name_column] == class_label].sample(frac=(1 - max_count / count)).index
            df = df.drop(remove_indices)
    return df


def save_csv(df, name):
    df.to_csv(name, index=False)


def heatmap(data):
    correlation_matrix = data.corr()
    plt.figure(figsize=(20, 13))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Тепловая карта корреляции')
    plt.show()




"""
numerical_graphics(dataCSV)
categorical_graphics(dataCSV)
search_passes(dataCSV)
replacing_passes(dataCSV)
search_passes(dataCSV)
"""

first_data = dataCSV.copy()
first_data = frequency_encoding(first_data)
first_data = minmax_normalization(first_data)

second_data = dataCSV.copy()
second_data = remove_rare(second_data)
second_data = balance_csv_dataset(second_data, "model")
second_data = balance_csv_dataset(second_data, "transmission")
second_data = balance_csv_dataset(second_data, "fuelType")
second_data = encoding_of_categorical_values(second_data)

third_data = dataCSV.copy()
third_data = remove_rare(third_data)
third_data = encoding_of_categorical_values(third_data)
# third_data = smote(third_data, "transmission")
# third_data = smote(third_data, "fuelType")
third_data = smote(third_data, "model")
third_data = minmax_normalization(third_data)

"""


"""

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

linear_params = {
    'fit_intercept': [True, False],
    'copy_X': [True, False],
    'n_jobs': [1, 2, 3, 4, 5, 6]
}

tree_params = {
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

kneighbors_params = {
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance'],
    'p': [1, 2]
}

ridge_params = {
    'alpha': [0.1, 1, 10]
}

randomForest_params = {
    'n_estimators': [50, 100],
    'max_features': ['sqrt'],
    'max_depth': [2, 7, 15],
    'min_samples_split': [3, 23],
    'min_samples_leaf': [3, 6],
}

gbm_params = {
    'n_estimators': [50, 100],
    'learning_rate': [0.1, 0.3],
    'max_depth': [3, 5]
}

adaBoost_params = {
    'n_estimators': [50, 100],
    'learning_rate': [0.1, 0.01, 0.001],
    'loss': ['linear', 'square', 'exponential']
}


def linear_regression(data, target_column, params):
    X = data.drop(columns=[target_column])
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = GridSearchCV(LinearRegression(), params, cv=5)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("Линейная регрессия")
    print('MAE:', mae)
    print('MSE:', mse)
    print('R2:', r2)


def decisionTreeRegressor(data, target_column, params):
    X = data.drop(columns=[target_column])
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = GridSearchCV(DecisionTreeRegressor(), params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("Дерево решений")
    print('MAE:', mae)
    print('MSE:', mse)
    print('R2:', r2)


def kNeighborsRegressor(data, target_column, params):
    X = data.drop(columns=[target_column])
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = GridSearchCV(KNeighborsRegressor(), params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("KNeighbors")
    print('MAE:', mae)
    print('MSE:', mse)
    print('R2:', r2)


def ridge(data, target_column, params):
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = GridSearchCV(Ridge(), params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("Ridge")
    print('MAE:', mae)
    print('MSE:', mse)
    print('R2:', r2)


def randomForest(data, target_column, params):
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = GridSearchCV(RandomForestRegressor(), params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("Случайный лес")
    print('MAE:', mae)
    print('MSE:', mse)
    print('R2:', r2)


def gradientBoosting(data, target_column, params):
    X = data.drop(columns=[target_column])
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = GridSearchCV(estimator=GradientBoostingRegressor(), param_grid=params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("Градиентный бустинг")
    print("MAE:", mae)
    print("MSE:", mse)
    print("R2:", r2)


def adaBoostRegression(data, target_column, params):
    X = data.drop(columns=[target_column])
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = GridSearchCV(AdaBoostRegressor(), params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("Ада бустинг")
    print("MAE:", mae)
    print("MSE:", mse)
    print("R2:", r2)

"""
start_time = time.time()
data_sets = [first_data, second_data, third_data]
data_names = ["Для первого набора данных", "Для второго набора данных", "Для третьего набора данных"]

for data, name in zip(data_sets, data_names):
    print(name)
    linear_regression(data, target_column, linear_params)
    decisionTreeRegressor(data, target_column, tree_params)
    kNeighborsRegressor(data, target_column, kneighbors_params)
    ridge(data, target_column, kneighbors_params)
    randomForest(data, target_column, randomForest_params)
    gradientBoosting(data, target_column, gbm_params)
    adaBoostRegression(data, target_column, adaBoost_params)

end_time = time.time()
execution_time = end_time - start_time
print(f"Время выполнения: {execution_time} секунд")
"""

ridge(first_data, target_column, ridge_params)
ridge(second_data, target_column, ridge_params)
ridge(third_data, target_column, ridge_params)