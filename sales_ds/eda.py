import numpy as np
import pandas as pd
from itertools import product




def build_monthly_grid(sales_train: pd.DataFrame) -> pd.DataFrame:
    monthly = (sales_train
               .groupby(['date_block_num', 'shop_id', 'item_id'])
               .agg(item_cnt_month=('item_cnt_day', 'sum'),
                    avg_price=('item_price', 'mean'))
               .reset_index())

    index_cols = ['shop_id', 'item_id', 'date_block_num']
    grid = []
    for block in range(34):
        cur_shops = monthly[monthly.date_block_num == block].shop_id.unique()
        cur_items = monthly[monthly.date_block_num == block].item_id.unique()
        grid.append(
            np.array(list(product([block], cur_shops, cur_items)),
                     dtype='int32').reshape(-1, 3)[:, [1, 2, 0]]
        )

    df = pd.DataFrame(np.vstack(grid), columns=index_cols)
    df = df.merge(monthly, on=index_cols, how='left').fillna(0)
    df['item_cnt_month'] = df['item_cnt_month'].clip(0, 20)
    return df



def add_lag_feature(df: pd.DataFrame, lags: list, col: str) -> pd.DataFrame:

    tmp = df[['date_block_num', 'shop_id', 'item_id', col]]
    for lag in lags:
        shifted = tmp.copy()
        shifted.columns = ['date_block_num', 'shop_id', 'item_id', f'{col}_lag_{lag}']
        shifted['date_block_num'] += lag
        df = df.merge(shifted, on=['date_block_num', 'shop_id', 'item_id'], how='left')
    return df




def add_group_averages(df: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:

    item_avg = (df.groupby(['date_block_num', 'item_id'])['item_cnt_month']
                .mean().reset_index()
                .rename(columns={'item_cnt_month': 'item_avg_cnt'}))
    df = df.merge(item_avg, on=['date_block_num', 'item_id'], how='left')
    df = add_lag_feature(df, [1, 2, 3], 'item_avg_cnt')


    shop_avg = (df.groupby(['date_block_num', 'shop_id'])['item_cnt_month']
                .mean().reset_index()
                .rename(columns={'item_cnt_month': 'shop_avg_cnt'}))
    df = df.merge(shop_avg, on=['date_block_num', 'shop_id'], how='left')
    df = add_lag_feature(df, [1, 2, 3], 'shop_avg_cnt')

    
    df = df.merge(items[['item_id', 'item_category_id']], on='item_id', how='left')

    
    cat_avg = (df.groupby(['date_block_num', 'item_category_id'])['item_cnt_month']
               .mean().reset_index()
               .rename(columns={'item_cnt_month': 'cat_avg_cnt'}))
    df = df.merge(cat_avg, on=['date_block_num', 'item_category_id'], how='left')
    df = add_lag_feature(df, [1], 'cat_avg_cnt')

    return df




def add_rolling_means(df: pd.DataFrame, windows: list = [3, 6, 12]) -> pd.DataFrame:
    """
    Скользящее среднее по лаг_1 за окно 3, 6, 12 месяцев.
    Сглаживает шум и показывает тренд продаж.
    """
    for win in windows:
        col = f'item_cnt_month_rmean_{win}'
        df[col] = (df.groupby(['shop_id', 'item_id'])['item_cnt_month_lag_1']
                   .rolling(win, min_periods=1)
                   .mean()
                   .reset_index(level=[0, 1], drop=True))
    return df




def add_trends(df: pd.DataFrame) -> pd.DataFrame:
  
    df['trend_1_2']  = df['item_cnt_month_lag_1'] - df['item_cnt_month_lag_2']
    df['trend_1_12'] = df['item_cnt_month_lag_1'] - df['item_cnt_month_lag_12']
    return df


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:

    df['month'] = df['date_block_num'] % 12
    df['year']  = df['date_block_num'] // 12
    return df


def build_test_features(test: pd.DataFrame, df: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
    
    test = test.copy()
    test["date_block_num"] = 34
    test["item_cnt_month"] = 0
    test["avg_price"]      = 0

    test = test.merge(items[["item_id", "item_category_id"]], on="item_id", how="left")
    test["month"] = 34 % 12
    test["year"]  = 34 // 12

    lag_source = df[["date_block_num", "shop_id", "item_id",
                     "item_cnt_month", "item_avg_cnt", "shop_avg_cnt",
                     "cat_avg_cnt", "avg_price", "item_category_id"]].copy()

    
    for lag in [1, 2, 3, 6, 12]:
        tmp = lag_source[lag_source["date_block_num"] == 34 - lag][
            ["shop_id", "item_id", "item_cnt_month", "avg_price"]
        ].rename(columns={
            "item_cnt_month": f"item_cnt_month_lag_{lag}",
            "avg_price": f"avg_price_lag_{lag}" if lag == 1 else "_drop"
        })
        if lag != 1:
            tmp = tmp.drop(columns=["_drop"])
        test = test.merge(tmp, on=["shop_id", "item_id"], how="left")


    item_data = lag_source[lag_source["date_block_num"].isin([33, 32, 31])].copy()
    item_data["lag"] = 34 - item_data["date_block_num"]
    item_pivot = (item_data
                  .pivot_table(index="item_id", columns="lag", values="item_avg_cnt")
                  .add_prefix("item_avg_cnt_lag_").reset_index())

    shop_data = lag_source[lag_source["date_block_num"].isin([33, 32, 31])].copy()
    shop_data["lag"] = 34 - shop_data["date_block_num"]
    shop_pivot = (shop_data
                  .pivot_table(index="shop_id", columns="lag", values="shop_avg_cnt")
                  .add_prefix("shop_avg_cnt_lag_").reset_index())

    test = test.merge(item_pivot, on="item_id", how="left")
    test = test.merge(shop_pivot, on="shop_id", how="left")

    
    test = add_lag_feature(test, [1], 'avg_price')

    cat_avg_33 = lag_source[lag_source["date_block_num"] == 33][
        ["item_category_id", "cat_avg_cnt"]
    ].drop_duplicates("item_category_id").rename(columns={"cat_avg_cnt": "cat_avg_cnt_lag_1"})
    test = test.merge(cat_avg_33, on="item_category_id", how="left")

    return test




def build_features(sales_train: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
    
    print('Строим месячную сетку...')
    df = build_monthly_grid(sales_train)

    print('Добавляем лаговые фичи item_cnt_month...')
    df = add_lag_feature(df, [1, 2, 3, 6, 12], 'item_cnt_month')

    print('Добавляем групповые средние...')
    df = add_group_averages(df, items)

    print('Добавляем лаг avg_price...')
    df = add_lag_feature(df, [1], 'avg_price')

    print('Добавляем скользящие средние...')
    df = add_rolling_means(df)

    print('Добавляем тренды...')
    df = add_trends(df)

    print('Добавляем временные фичи...')
    df = add_date_features(df)

    print('Feature engineering завершён')
    return df