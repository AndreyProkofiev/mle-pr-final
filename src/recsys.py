import os
import sys
import pandas as pd
from lightfm import LightFM
from lightfm.data import Dataset
from lightfm.evaluation import precision_at_k
from lightfm.data import Dataset
from lightfm.evaluation import precision_at_k

def prepare_data(df):
    """
    Подготавливает данные для обучения модели LightFM.
    Создает user-item взаимодействия и веса на основе рейтингов.
    """
    dataset = Dataset()
    dataset.fit(df['visitorid'], df['itemid'])
    
    # Создаем матрицу взаимодействий с весами (рейтингами)
    (interactions, weights) = dataset.build_interactions(
        [(row['visitorid'], row['itemid'], row['rating']) for _, row in df.iterrows()]
    )
    
    # Получаем маппинг пользователей и айтемов
    user_id_map = dataset.mapping()[0]
    item_id_map = dataset.mapping()[2]
    
    return interactions, weights, user_id_map, item_id_map


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


def recommend_items(model, user_id, user_id_map, item_id_map, interactions, top_n=10):
    """
    Делает рекомендации для конкретного пользователя.
    """
    if user_id not in user_id_map:
        raise ValueError(f"User ID {user_id} not found in the dataset.")
    
    user_idx = user_id_map[user_id]
    scores = model.predict(user_idx, np.arange(len(item_id_map)))
    
    # Исключаем уже взаимодействовавшие айтемы
    known_items = interactions.tocsr()[user_idx].indices
    scores[known_items] = -np.inf
    
    # Выбираем топ-N айтемов
    top_items = np.argsort(-scores)[:top_n]
    top_item_ids = [list(item_id_map.keys())[list(item_id_map.values()).index(i)] for i in top_items]
    
    return top_item_ids


def get_user_embeddings(model, user_id_map):
    """
    Возвращает эмбеддинги всех пользователей.
    """
    user_embeddings = model.user_embeddings
    user_ids = list(user_id_map.keys())
    user_embeddings_dict = {user_id: user_embeddings[user_id_map[user_id]] for user_id in user_ids}
    return user_embeddings_dict


def recommend_items_score(model, user_id, user_id_map, item_id_map, interactions, top_n=10):
    """
    Делает рекомендации для конкретного пользователя и возвращает их в виде датафрейма.
    
    Возвращает:
        pd.DataFrame: DataFrame с двумя колонками: 'itemid' (ID айтема) и 'score' (скор).
    """
    if user_id not in user_id_map:
        raise ValueError(f"User ID {user_id} not found in the dataset.")
    
    user_idx = user_id_map[user_id]
    scores = model.predict(user_idx, np.arange(len(item_id_map)))
    
    # Исключаем уже взаимодействовавшие айтемы
    known_items = interactions.tocsr()[user_idx].indices
    scores[known_items] = -np.inf
    
    # Выбираем топ-N айтемов и их скоры
    top_indices = np.argsort(-scores)[:top_n]
    top_scores = scores[top_indices]
    top_item_ids = [list(item_id_map.keys())[list(item_id_map.values()).index(i)] for i in top_indices]
    
    # Создаем DataFrame
    recommendations_df = pd.DataFrame({
        'itemid': top_item_ids,
        'score': top_scores
    })
    
    return recommendations_df


def predict_for_new_user(model, dataset, new_user_features, item_id_map, top_n=10):
    """
    Делает рекомендации для нового пользователя на основе его признаков.
    """
    # Создаем разреженную матрицу признаков для нового пользователя
    (new_user_features_matrix, _) = dataset.build_user_features([new_user_features])
    
    # Предсказываем оценки для всех айтемов
    scores = model.predict(0, np.arange(len(item_id_map)), user_features=new_user_features_matrix)
    
    # Выбираем топ-N айтемов
    top_items = np.argsort(-scores)[:top_n]
    top_item_ids = [list(item_id_map.keys())[list(item_id_map.values()).index(i)] for i in top_items]
    
    return top_item_ids

def recommend_for_cold_user(model, dataset, user_features, item_id_map, top_n=10):
    """
    Делает рекомендации для нового (холодного) пользователя на основе его признаков.
    
    Параметры:
        model: Обученная модель LightFM.
        dataset: Экземпляр Dataset, используемый для подготовки данных.
        user_features: Список или словарь признаков нового пользователя.
        item_id_map: Словарь маппинга ID айтемов.
        top_n: Количество рекомендаций для выдачи.
    
    Возвращает:
        pd.DataFrame: DataFrame с двумя колонками: 'itemid' (ID айтема) и 'score' (скор).
    """
    # Преобразуем признаки пользователя в формат, подходящий для LightFM
    if isinstance(user_features, dict):
        user_features = [(key, value) for key, value in user_features.items()]
    
    # Создаем разреженную матрицу признаков для нового пользователя
    (user_features_matrix, _) = dataset.build_user_features([user_features])
    
    # Предсказываем оценки для всех айтемов
    scores = model.predict(0, np.arange(len(item_id_map)), user_features=user_features_matrix)
    
    # Выбираем топ-N айтемов и их скоры
    top_indices = np.argsort(-scores)[:top_n]
    top_scores = scores[top_indices]
    top_item_ids = [list(item_id_map.keys())[list(item_id_map.values()).index(i)] for i in top_indices]
    
    # Создаем DataFrame
    recommendations_df = pd.DataFrame({
        'itemid': top_item_ids,
        'score': top_scores
    })
    
    return recommendations_df