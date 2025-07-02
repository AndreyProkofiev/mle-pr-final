import os
import sys
import pandas as pd
from lightfm import LightFM
from lightfm.data import Dataset
from lightfm.evaluation import precision_at_k
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


def train_model(interactions, weights=None, epochs=30, num_threads=4, model_params=None):
    """
    Обучает модель LightFM с использованием весов (рейтингов) и пользовательских параметров.
    
    Параметры:
        interactions: Матрица взаимодействий (разреженная матрица).
        weights: Матрица весов (разреженная матрица), опционально.
        epochs: Количество эпох обучения.
        num_threads: Количество потоков для параллельного обучения.
        model_params: Словарь параметров для модели LightFM (например, loss, learning_rate и т.д.).
                      Если None, используются параметры по умолчанию.
    
    Возвращает:
        model: Обученная модель LightFM.
    """
    # Параметры по умолчанию
    default_params = {
        'loss': 'warp',  # Функция потерь (например, 'warp', 'bpr', 'logistic')
        'learning_rate': 0.05,
        'item_alpha': 0.0,
        'user_alpha': 0.0,
        'no_components': 10  # Размерность эмбеддингов
    }
    
    # Объединяем параметры по умолчанию с пользовательскими параметрами
    if model_params is not None:
        default_params.update(model_params)
    
    # Создаем модель LightFM с указанными параметрами
    model = LightFM(**default_params)
    
    # Обучаем модель
    model.fit(
        interactions,
        sample_weight=weights,
        epochs=epochs,
        num_threads=num_threads
    )
    
    return model

def calculate_precision_at_k(model, train_interactions, test_interactions, k=10, num_threads=4):
    """
    Рассчитывает среднюю precision@k для тестовой выборки.
    
    Параметры:
        model: Обученная модель LightFM.
        train_interactions: Матрица взаимодействий обучающей выборки.
        test_interactions: Матрица взаимодействий тестовой выборки.
        k: Количество рекомендаций для оценки (top-K).
        num_threads: Количество потоков для параллельного вычисления.
    
    Возвращает:
        float: Среднее значение precision@k для всех пользователей в тестовой выборке.
    """
    # Вычисляем precision@k
    precision = precision_at_k(
        model=model,
        test_interactions=test_interactions,
        train_interactions=train_interactions,
        k=k,
        num_threads=num_threads
    ).mean()
    
    return precision

