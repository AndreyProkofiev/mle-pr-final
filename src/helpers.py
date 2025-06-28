import subprocess
import os
import json


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

