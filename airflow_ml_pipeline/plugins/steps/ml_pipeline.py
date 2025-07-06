###ml_pipeline.py
# from steps.ml_pipeline import prepare_data_with_split, grid_search_hyperparameters, save_model
import os
import sys
import numpy as np
import pandas as pd
import pickle
from lightfm import LightFM
from itertools import product
from lightfm.data import Dataset
from lightfm.evaluation import precision_at_k



def prepare_data_with_split(df, test_size=0.2):
    dataset = Dataset()
    dataset.fit(df['visitorid'], df['itemid'])
    
    train_df = df.sample(frac=1 - test_size, random_state=42)
    test_df = df.drop(train_df.index)
    
    # Фильтрация тестовой выборки
    test_df = filter_cold_users_items(train_df, test_df)
    
    train_interactions, _ = dataset.build_interactions(
        [(row['visitorid'], row['itemid']) for _, row in train_df.iterrows()]
    )
    test_interactions, _ = dataset.build_interactions(
        [(row['visitorid'], row['itemid']) for _, row in test_df.iterrows()]
    )
    
    return train_interactions, test_interactions, dataset


# Фильтрация холодных пользователей и айтемов
def filter_cold_users_items(train_df, test_df):
    train_users = set(train_df['visitorid'])
    train_items = set(train_df['itemid'])
    
    test_df = test_df[
        (test_df['visitorid'].isin(train_users)) &
        (test_df['itemid'].isin(train_items))
    ]
    
    return test_df

def evaluate_model(train_interactions, test_interactions, params, k=5):
    """
    Обучает модель LightFM и оценивает её качество с помощью precision@k.
    
    Параметры:
        interactions: Матрица взаимодействий.
        params: Словарь параметров модели.
        k: Количество рекомендаций для оценки метрики precision@k.
    
    Возвращает:
        float: Значение precision@k.
    """
    # Разделение данных на обучающую и тестовую выборки
    # train_interactions, test_interactions, dataset = prepare_data_with_split(train_df)
    
    # Обучение модели
    model_params = {k:v for k,v in params.items() if k not in 'epochs'}
    model = LightFM(**model_params)
    model.fit(train_interactions, epochs=params.get('epochs', 30), num_threads=4)
    
    # Оценка модели
    precision = precision_at_k(
        model=model,
        test_interactions=test_interactions,
        train_interactions=train_interactions,
        k=k,
        num_threads=4
    ).mean()
    
    return precision



def grid_search_hyperparameters(train_df, param_grid, k=5):
    """
    Выполняет подбор гиперпараметров с помощью Grid Search.
    
    Параметры:
        interactions: Матрица взаимодействий.
        param_grid: Словарь с сеткой гиперпараметров.
        k: Количество рекомендаций для оценки метрики precision@k.
    
    Возвращает:
        dict: Лучшие параметры и соответствующее значение precision@k.
    """
    grid_search_m3x = []
    best_params = None
    best_score = -1
    
    # Перебор всех комбинаций гиперпараметров
    train_interactions, test_interactions, dataset = prepare_data_with_split(train_df)
    keys, values = zip(*param_grid.items())
    for params_values in product(*values):
        params = dict(zip(keys, params_values))
        params_df = pd.DataFrame([params])
        # print(f"Тестируем параметры: {params}")
        
        # Оценка модели
        score = evaluate_model(train_interactions, test_interactions, params, k)
        params_df[f'Precision@{k}'] = score
        # print(f"Precision@{k}: {score:.4f}")
        grid_search_m3x.append(params_df)
        print(params_df)
        
        # Сохранение лучших параметров
        if score > best_score:
            best_score = score
            best_params = params
    
    print(f"\nЛучшие параметры: {best_params}")
    print(f"Лучший precision@{k}: {best_score:.4f}")
    log_search = pd.concat(grid_search_m3x,ignore_index=True) 

    
    return best_params, best_score, log_search 


def save_model(model, filepath):
    """
    Сохраняет обученную модель LightFM на диск.
    
    Параметры:
        model: Обученная модель LightFM.
        filepath: Путь к файлу, куда будет сохранена модель.
    """
    with open(filepath, 'wb') as file:
        pickle.dump(model, file)
    return print(f"Модель успешно сохранена в файл: {filepath}")


