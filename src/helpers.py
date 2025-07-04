import subprocess
import os
import sys
import json
import numpy as np
import pandas as pd
import mlflow
import pickle
from lightfm import LightFM
from lightfm.evaluation import precision_at_k
sys.path.append('/home/mle-user/mle_projects/mle-pr-final/')
from src.recsys import prepare_data_with_split, train_model, calculate_precision_at_k, load_model


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


def log_lightfm_model_to_mlflow(model_filepath, model_params, train_interactions, test_interactions, k=5):
    """
    Логгирует модель LightFM в MLflow.
    
    Параметры:
        model: Обученная модель LightFM.
        model_params: Словарь параметров модели (например, loss, learning_rate и т.д.).
        train_interactions: Матрица взаимодействий обучающей выборки.
        test_interactions: Матрица взаимодействий тестовой выборки.
        k: Количество рекомендаций для оценки метрики precision@k.
    """
    model = load_model(model_filepath)
    # Начинаем эксперимент MLflow
    with mlflow.start_run():
        # 1. Логгирование параметров модели
        mlflow.log_params(model_params)
        
        # 2. Расчет метрики precision@k
        precision = precision_at_k(
            model=model,
            test_interactions=test_interactions,
            train_interactions=train_interactions,
            k=k,
            num_threads=4
        ).mean()
        
        # Логгирование метрики
        mlflow.log_metric("precision_at_k", precision)
        
        # 3. Сохранение модели в файл
        model_filepath = "lightfm_model.pkl"
        with open(model_filepath, 'wb') as file:
            pickle.dump(model, file)
        
        # 4. Логгирование модели как артефакта
        mlflow.log_artifact(model_filepath, artifact_path="model")


def mk_model_data(conf_path: str):
    config = load_config(conf_path)
    raw_data_path = config['raw_data_path']
    event_df  = pd.read_csv(os.path.join(raw_data_path, 'events.csv'))
    model_df = event_df[event_df['event'] != 'transaction'].copy()
    model_df['rating'] = np.where(model_df['event'] == 'addtocart', 5,1)
    filter_users = event_df.groupby('visitorid', as_index=0)['itemid'].nunique()
    train_users = filter_users[(filter_users['itemid']>10)].visitorid.unique()
    cond_tr = (model_df['visitorid'].isin(train_users))
    train_df = model_df[cond_tr][['visitorid', 'itemid', 'rating']]
    train_df = train_df.groupby(['visitorid','itemid'], as_index=0).rating.max()
    ##To DO del dataframes
    return train_df

        
       