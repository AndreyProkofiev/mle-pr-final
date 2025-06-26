import subprocess
import os
import json


def load_config_to_globals(config_path):
    """
    Загружает данные из JSON-файла и создает глобальные переменные
    с именами и значениями, указанными в конфигурации.

    :param config_path: Путь к JSON-файлу с конфигурацией
    """
    # Открываем и читаем JSON-файл
    with open(config_path, "r") as json_file:
        config = json.load(json_file)

    # Извлечение данных из конфигурации и объявление глобальных переменных
    global EXPERIMENT_NAME, RUN_NAME, REGISTRY_MODEL_NAME, model_path, data_path, metrics_path, requirements_path

    EXPERIMENT_NAME = config.get("experiment_name", "default_experiment")
    RUN_NAME = config.get("run_name", "default_run")
    REGISTRY_MODEL_NAME = config.get("registry_model_name", "default_model")
    model_path = config.get("model_path")
    data_path = config.get("data_path")
    metrics_path = config.get("metrics_path")
    requirements_path = config.get("requirements_path")
    recsys_dir = config.get("recsys_path")



def start_mlflow_server(config_path):
    """
    Функция для запуска MLflow сервера через bash-скрипт.
    Выполняет переход в директорию recsys и запуск run_mlflow_server.sh.
    """
    with open(config_path, "r") as json_file:
        config = json.load(json_file)
    recsys_dir = config.get("recsys_path")
    
    try:
        # Переход в директорию recsys
        # recsys_dir = os.path.join(os.getcwd(), "recsys")
        os.chdir(recsys_dir)
        print(f"Перешли в директорию: {recsys_dir}")

        # Запуск bash-скрипта
        script_path = "run_mlflow_server.sh"
        print(f"Запуск скрипта: {script_path}")
        result = subprocess.run(["sh", script_path], check=True, text=True, capture_output=True)

        # Вывод результата выполнения
        print("Сервер успешно запущен!")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)

    except FileNotFoundError:
        print("Ошибка: Директория recsys или файл run_mlflow_server.sh не найдены.")
    except subprocess.CalledProcessError as e:
        print(f"Ошибка при выполнении скрипта: {e}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
    except Exception as e:
        print(f"Произошла ошибка: {e}")

# Пример использования
if __name__ == "__main__":
    start_mlflow_server()