###train_lightfm_model.py
import pendulum
from airflow.decorators import dag, task
from steps.send_telegram import  send_telegram_success_message, send_telegram_failure_message

@dag(
    schedule='@once',
    start_date=pendulum.datetime(2025, 1, 1, tz="UTC"),
    tags=["ML_pipeline"],
    catchup=False,
    on_success_callback=send_telegram_success_message,
    on_failure_callback=send_telegram_failure_message
)
def run_ml_pipeline():
    import pandas as pd
    import numpy as np
    from airflow.providers.postgres.hooks.postgres import PostgresHook
    from sqlalchemy import MetaData, Table, Column, String, Integer, Float, DateTime, UniqueConstraint,inspect

    @task()
   
    @task()
    def extract(**kwargs):
        """
        #### Extract task
        """
        hook = PostgresHook('destination_db')
        conn = hook.get_conn()
        sql = f"""select * from ecom_events"""
        data = pd.read_sql(sql, conn)
        conn.close()
        return data

    @task()
    def mk_model_data(data: pd.DataFrame):
  
        model_df = data[data['event'] != 'transaction'].copy()
        model_df['rating'] = np.where(model_df['event'] == 'addtocart', 5,1)
        filter_users = data.groupby('visitorid', as_index=0)['itemid'].nunique()
        train_users = filter_users[(filter_users['itemid']>10)].visitorid.unique()
        cond_tr = (model_df['visitorid'].isin(train_users))
        train_df = model_df[cond_tr][['visitorid', 'itemid', 'rating']]
        train_df = train_df.groupby(['visitorid','itemid'], as_index=0).rating.max()
        ##To DO del dataframes
        return train_df

    @task()
    def train_model(data: pd.DataFrame):
        """
        #### Load task
        """
        from steps.ml_pipeline import prepare_data_with_split, grid_search_hyperparameters, save_model
        param_grid = {
        'loss': ['warp', 'bpr'],
        'learning_rate': [0.01, 0.05],
        'no_components': [10, 50, 150],
        'epochs': [30, 50, 101] }
        best_params, best_score, log_search = grid_search_hyperparameters(data, param_grid, k=5)
        train_interactions, test_interactions, dataset = prepare_data_with_split(data)
        model_params = {k:v for k,v in best_params.items() if k not in 'epochs'}
        model_best_params_gs = train_model(train_interactions, epochs=best_params.get('epochs', 30), model_params=model_params)

        save_model(model_best_params_gs, "model_best_params_gs.pkl")
        log_search.to_csv('log_search.csv')
    

    
    data = extract()
    transformed_data = mk_model_data(data)
    train_model(transformed_data)

run_ml_pipeline()

