###recommendations_service.py
import requests
from requests.exceptions import ConnectionError
from fastapi import FastAPI
from prometheus_client import Counter, Gauge
from prometheus_fastapi_instrumentator import Instrumentator
from lightfm import LightFM
from lightfm.data import Dataset
import os
import pandas as pd
import random
import numpy as np
from typing import List, Optional
import sys
from helpers import load_config, log_lightfm_model_to_mlflow, mk_model_data
from src.recsys import  load_model,  prepare_data, recommend_items

app = FastAPI()

###
conf_path = '/home/mle-user/mle_projects/mle-pr-final/config/config.json'
config = load_config(conf_path)
model_path = config['model_path']
model = load_model(os.path.join(model_path, 'model_best_params_gs.pkl'))
light_df = mk_model_data(conf_path)
interactions, weights, user_id_map, item_id_map, dataset = prepare_data(light_df)

@app.get("/recommendations/{user_id}")
def get_recommendations(user_id: str):
    try:
        user_id_int = int(user_id)
    except ValueError:
        return {"error": "Invalid user ID format"}

    try:
        recs = recommend_items(model, user_id_int, user_id_map, item_id_map, interactions, top_n=10)

        return {
            "user_id": user_id_int,
            "recommendations": recs
        }
    except Exception as e:
        return {"error": f"Internal error: {str(e)}"}
    