import gc
import numpy as np
import optuna
from catboost import CatBoostRegressor, Pool


def expanding_optuna(
    df,
    val_blocks: list,
    feature_cols: list,
    start_block: int = 12,
    n_trials: int = 10,
) -> tuple:
   
    cat_features = [c for c in ["shop_id", "item_id", "item_category_id"]
                    if c in feature_cols]

    total = n_trials * len(val_blocks)
    print(f"Категориальные фичи: {cat_features}")
    print(f"Трайлов: {n_trials}, блоков: {len(val_blocks)}, обучений: {total}")
    print(f"Оценка времени: ~{total * 3}–{total * 3 + 10} минут\n")

    def objective(trial):
        params = {
            "iterations": 300,
            "learning_rate": trial.suggest_float("learning_rate", 0.06, 0.15, log=True),
            "depth": trial.suggest_int("depth", 5, 7),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.5, 10.0, log=True),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 150),
            "bootstrap_type": "Bernoulli",
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "random_seed": 11,
            "eval_metric": "RMSE",
            "loss_function": "RMSE",
            "early_stopping_rounds": 40,
            "use_best_model": True,
            #"task_type": "GPU",
            "devices": "0",
            "verbose": 0,
        }

        cv_scores = []

        for val_block in val_blocks:
            train_mask = (
                (df.date_block_num >= start_block) &
                (df.date_block_num < val_block)
            )
            val_mask = (df.date_block_num == val_block)

            X_tr  = df.loc[train_mask, feature_cols].copy()
            X_val = df.loc[val_mask,   feature_cols].copy()

            for col in cat_features:
                X_tr[col]  = X_tr[col].astype(str)
                X_val[col] = X_val[col].astype(str)

            train_pool = Pool(
                data=X_tr,
                label=df.loc[train_mask, "item_cnt_month"],
                cat_features=cat_features
            )
            val_pool = Pool(
                data=X_val,
                label=df.loc[val_mask, "item_cnt_month"],
                cat_features=cat_features
            )

            del X_tr, X_val
            gc.collect()

            model = CatBoostRegressor(**params)
            model.fit(train_pool, eval_set=val_pool)
            cv_scores.append(model.get_best_score()["validation"]["RMSE"])

            del train_pool, val_pool, model
            gc.collect()

        
        if len(cv_scores) == 2:
            score = cv_scores[0] * 0.4 + cv_scores[1] * 0.6
        else:
            score = float(np.mean(cv_scores))

        trial.set_user_attr("scores_per_block", cv_scores)
        return score

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        direction="minimize",
        study_name="CatBoost_CV",
        sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=4)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\nЛучший RMSE: {study.best_value:.4f}")
    print("Лучшие параметры:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    best_scores = study.best_trial.user_attrs.get("scores_per_block", [])
    for block, score in zip(val_blocks, best_scores):
        print(f"  block {block}: RMSE = {score:.4f}")

    return study.best_params, study


def build_final_params(best_params: dict, iterations: int = 500) -> dict:
    
    final_params = best_params.copy()
    final_params.update({
        "iterations": iterations,
        "random_seed": 11,
        "bootstrap_type": "Bernoulli",
        "eval_metric": "RMSE",
        "loss_function": "RMSE",
        "use_best_model": False,
        #"task_type": "GPU",
        "devices": "0",
        "verbose": 100,
    })
    final_params.pop("early_stopping_rounds", None)
    return final_params