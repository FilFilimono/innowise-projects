"""
Пайплайн:
  1. extract_data   — читает raw_data, запускает DQC → clean_data
  2. run_fe         — feature engineering → df + test с фичами
  3. run_model      — загружает модель из wandb, делает предсказания
  4. store_analysis — сохраняет CSV, логирует статистику в wandb
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

dag = DAG(
    dag_id="sales_pipeline",
    description="Predict Future Sales — полный инференс пайплайн",
    schedule_interval=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    default_args=default_args,
)

DATA_PATH    = "/opt/airflow/data"
RESULTS_PATH = "/opt/airflow/results"


def extract_data(**context):
   
    from sales_ds.dqc import etl_layer

    etl_layer(
        raw_data_path=f"{DATA_PATH}/raw_data",
        clean_data_path=f"{DATA_PATH}/clean_data",
        json_path=f"{DATA_PATH}/json/issues_summary.json",
    )
    print("✓ DQC + ETL завершён")



def run_fe(**context):
    import pandas as pd
    from sales_ds.eda import build_features

    sales_train    = pd.read_csv(f"{DATA_PATH}/clean_data/sales_train.csv")
    items          = pd.read_csv(f"{DATA_PATH}/clean_data/items.csv")
    test_raw       = pd.read_csv(f"{DATA_PATH}/test/test.csv")

    df, test = build_features(sales_train, items, test_raw)

    df.to_parquet("/tmp/df_features.parquet", index=False)
    test.to_parquet("/tmp/test_features.parquet", index=False)

    print(f"✓ FE завершён: df={df.shape}, test={test.shape}")



def run_model(**context):
    
    import os
    import pandas as pd
    import wandb
    from catboost import CatBoostRegressor

    test = pd.read_parquet("/tmp/test_features.parquet")

    drop_cols = [
        "item_cnt_month", "date_block_num", "avg_price",
        "item_avg_cnt", "shop_avg_cnt", "cat_avg_cnt", "cat_month_avg",
        "cat_last_vs_mean", "cat_trend_3m", "year",
        "item_cnt_month_rmean_12", "item_cnt_month_lag_12",
        "item_active_months", "item_lag1_global", "trend_1_2",
        "cat_avg_cnt_lag_1", "cat_avg_cnt",
    ]
    feature_cols = [c for c in test.columns if c not in drop_cols + ["ID"]]

    cat_features = ["shop_id", "item_id", "item_category_id"]
    X_test = test[feature_cols].copy()
    for col in cat_features:
        if col in X_test.columns:
            X_test[col] = X_test[col].astype(str)

    run = wandb.init(
        project="predict-future-sales",
        job_type="inference",
        name=f"inference-{datetime.now().strftime('%Y%m%d-%H%M')}",
    )
    artifact = run.use_artifact("catboost-model:latest", type="model")
    artifact_dir = artifact.download()

    model = CatBoostRegressor()
    model.load_model(f"{artifact_dir}/cat_model.cbm")

    predictions = model.predict(X_test).clip(0, 20)

    result = pd.DataFrame({
        "ID": test["ID"],
        "item_cnt_month": predictions
    })
    result.to_csv("/tmp/predictions.csv", index=False)

    context["ti"].xcom_push(key="wandb_run_id", value=run.id)
    run.finish()
    print(f"✓ Предсказания: {len(predictions)} строк")



def store_analysis(**context):
    import os
    import pandas as pd
    import wandb

    df = pd.read_csv("/tmp/predictions.csv")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_path = f"{RESULTS_PATH}/predictions_{timestamp}.csv"
    df.to_csv(output_path, index=False)
    print(f"✓ Сохранено: {output_path}")

    stats = {
        "total_predictions": len(df),
        "mean_prediction":   float(df["item_cnt_month"].mean()),
        "std_prediction":    float(df["item_cnt_month"].std()),
        "zero_predictions":  int((df["item_cnt_month"] == 0).sum()),
        "max_prediction":    float(df["item_cnt_month"].max()),
        "pct_zero":          float((df["item_cnt_month"] == 0).mean() * 100),
    }
    for k, v in stats.items():
        print(f"  {k}: {v}")

    run = wandb.init(
        project="predict-future-sales",
        job_type="analysis",
        name=f"analysis-{timestamp}",
    )
    wandb.log(stats)

    artifact = wandb.Artifact("predictions", type="dataset")
    artifact.add_file(output_path)
    run.log_artifact(artifact)

    run.finish()
    print("✓ Анализ залогирован в wandb")



task_extract = PythonOperator(task_id="extract_data",   python_callable=extract_data,  dag=dag)
task_fe      = PythonOperator(task_id="run_fe",         python_callable=run_fe,        dag=dag)
task_model   = PythonOperator(task_id="run_model",      python_callable=run_model,     dag=dag)
task_store   = PythonOperator(task_id="store_analysis", python_callable=store_analysis, dag=dag)

task_extract >> task_fe >> task_model >> task_store