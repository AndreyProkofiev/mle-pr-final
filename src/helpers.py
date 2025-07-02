import subprocess
import os
import json
import pandas as pd
from lightfm import LightFM
from lightfm.data import Dataset
from lightfm.evaluation import precision_at_k
from lightfm.data import Dataset
from lightfm.evaluation import precision_at_k


def start_mlflow_server(config_path):
    """
    Функция для запуска MLflow сервера через bash-скрипт.
    Сервер запускается в фоновом режиме, чтобы не блокировать выполнение ячейки.
    PID процесса сохраняется в файл для последующей остановки.
    """
    # Чтение конфигурации
    with open(config_path, "r") as json_file:
        config = json.load(json_file)
    mlfl_dir = config.get("mlflow_dir_path")
    
    try:
        # Переход в директорию recsys
        os.chdir(mlfl_dir)
        print(f"Перешли в директорию: {mlfl_dir}")

        # Путь к скрипту
        script_path = "run_mlflow_server.sh"
        if not os.path.isfile(script_path):
            raise FileNotFoundError(f"Файл {script_path} не найден в директории {mlfl_dir}")

        # Запуск скрипта в фоновом режиме
        process = subprocess.Popen(["sh", script_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Сервер запущен в фоновом режиме. PID: {process.pid}")

        # Сохранение PID в файл для последующей остановки
        pid_file = os.path.join(mlfl_dir, "mlflow_server.pid")
        with open(pid_file, "w") as f:
            f.write(str(process.pid))
        print(f"PID сервера сохранен в файл: {pid_file}")

    except FileNotFoundError as e:
        print(f"Ошибка: {e}")
    except Exception as e:
        print(f"Произошла ошибка: {e}")
        
    
def stop_mlflow_server(config_path):
    """
    Функция для остановки MLflow сервера.
    Использует PID, сохраненный при запуске сервера.
    """
    # Чтение конфигурации
    with open(config_path, "r") as json_file:
        config = json.load(json_file)
    mlfl_dir = config.get("mlflow_dir_path")
    
    try:
        # Путь к файлу с PID
        pid_file = os.path.join(mlfl_dir, "mlflow_server.pid")
        if not os.path.isfile(pid_file):
            raise FileNotFoundError(f"Файл с PID ({pid_file}) не найден.")

        # Чтение PID из файла
        with open(pid_file, "r") as f:
            pid = int(f.read().strip())
        print(f"Найден PID сервера: {pid}")

        # Попытка завершить процесс
        try:
            os.kill(pid, 9)  # SIGKILL для принудительного завершения
            print(f"Сервер с PID {pid} успешно остановлен.")
        except ProcessLookupError:
            print(f"Процесс с PID {pid} уже завершен или не существует.")

        # Удаление файла с PID
        os.remove(pid_file)
        print(f"Файл с PID удален: {pid_file}")

    except FileNotFoundError as e:
        print(f"Ошибка: {e}")
    except Exception as e:
        print(f"Произошла ошибка: {e}")



def load_config(config_path):
 
    with open(config_path, "r") as json_file:
        config = json.load(json_file)

    return config



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