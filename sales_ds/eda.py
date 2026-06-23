

import numpy as np
import pandas as pd
from itertools import product


def build_monthly_grid(sales_train: pd.DataFrame) -> pd.DataFrame:
    
    monthly = sales_train.groupby(
        ['date_block_num', 'shop_id', 'item_id']
    ).agg(
        item_cnt_month=('item_cnt_day', 'sum'),
        avg_price=('item_price', 'mean')
    ).reset_index()

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


def add_extra_features(df: pd.DataFrame, test: pd.DataFrame) -> tuple:
    
    item_lag1_global = (df[df.date_block_num == 33]
                        .groupby('item_id')['item_cnt_month']
                        .sum().reset_index()
                        .rename(columns={'item_cnt_month': 'item_lag1_global'}))
    item_active = (df[df.date_block_num.isin([31, 32, 33])]
                   .groupby('item_id')['item_cnt_month']
                   .apply(lambda x: (x > 0).sum()).reset_index()
                   .rename(columns={'item_cnt_month': 'item_active_months'}))

    df   = df.merge(item_lag1_global, on='item_id', how='left')
    df   = df.merge(item_active, on='item_id', how='left')
    test = test.merge(item_lag1_global, on='item_id', how='left')
    test = test.merge(item_active, on='item_id', how='left')

    
    item_stats = (df.groupby(['item_id', 'date_block_num'])['item_cnt_month']
                  .mean().reset_index().sort_values('date_block_num'))
    item_stats['item_max_sales'] = (item_stats.groupby('item_id')['item_cnt_month']
                                    .transform(lambda x: x.shift(1).expanding().max()))
    item_stats['item_q90_sales'] = (item_stats.groupby('item_id')['item_cnt_month']
                                    .transform(lambda x: x.shift(1).expanding().quantile(0.9)))
    item_stats = item_stats[['item_id', 'date_block_num', 'item_max_sales', 'item_q90_sales']]

    df = df.merge(item_stats, on=['item_id', 'date_block_num'], how='left')
    item_stats_test = item_stats[item_stats.date_block_num == 33][
        ['item_id', 'item_max_sales', 'item_q90_sales']]
    test = test.merge(item_stats_test, on='item_id', how='left')

    
    lag_cols = ['item_cnt_month_lag_1', 'item_cnt_month_lag_2', 'item_cnt_month_lag_3']
    df['was_sold_last_3m']   = df[lag_cols].gt(0).any(axis=1).astype(int)
    test['was_sold_last_3m'] = test[lag_cols].gt(0).any(axis=1).astype(int)

    
    cat_month_stats = (df.groupby(['item_category_id', 'date_block_num', 'month'])
                       ['item_cnt_month'].mean().reset_index().sort_values('date_block_num'))
    cat_month_stats['cat_month_avg'] = (
        cat_month_stats.groupby(['item_category_id', 'month'])['item_cnt_month']
        .transform(lambda x: x.shift(1).expanding().mean()))
    cat_month_stats = cat_month_stats[['item_category_id', 'date_block_num', 'cat_month_avg']]
    df = df.merge(cat_month_stats, on=['item_category_id', 'date_block_num'], how='left')
    cat_month_test = cat_month_stats[cat_month_stats.date_block_num == 33][
        ['item_category_id', 'cat_month_avg']]
    test = test.merge(cat_month_test, on='item_category_id', how='left')

    
    cat_dyn = (df.groupby(['item_category_id', 'date_block_num'])['item_cnt_month']
               .mean().reset_index().sort_values('date_block_num'))
    cat_dyn['cat_trend_3m'] = (cat_dyn.groupby('item_category_id')['item_cnt_month']
                                .transform(lambda x: x.shift(1).diff(3)))
    cat_dyn = cat_dyn[['item_category_id', 'date_block_num', 'cat_trend_3m']]
    df = df.merge(cat_dyn, on=['item_category_id', 'date_block_num'], how='left')
    cat_trend_test = cat_dyn[cat_dyn.date_block_num == 33][['item_category_id', 'cat_trend_3m']]
    test = test.merge(cat_trend_test, on='item_category_id', how='left')

    
    cat_historical = (df[df.date_block_num < 32].groupby('item_category_id')['item_cnt_month']
                      .mean().reset_index()
                      .rename(columns={'item_cnt_month': 'cat_historical_mean'}))
    cat_last = (df[df.date_block_num == 32].groupby('item_category_id')['item_cnt_month']
                .mean().reset_index()
                .rename(columns={'item_cnt_month': 'cat_last_mean'}))
    cat_ratio = cat_historical.merge(cat_last, on='item_category_id', how='left')
    cat_ratio['cat_last_vs_mean'] = cat_ratio['cat_last_mean'] / (cat_ratio['cat_historical_mean'] + 1)
    cat_ratio = cat_ratio[['item_category_id', 'cat_last_vs_mean']]
    df   = df.merge(cat_ratio, on='item_category_id', how='left')
    test = test.merge(cat_ratio, on='item_category_id', how='left')

    return df, test


def build_test_features(test_raw: pd.DataFrame, df: pd.DataFrame,
                        items: pd.DataFrame) -> pd.DataFrame:
    """Строит лаговые фичи для тест сета (block 34) из трейн данных."""
    test = test_raw.copy()
    test['date_block_num'] = 34
    test['item_cnt_month'] = 0
    test['avg_price']      = 0
    test = test.merge(items[['item_id', 'item_category_id']], on='item_id', how='left')
    test['month'] = 34 % 12
    test['year']  = 34 // 12

    lag_source = df[['date_block_num', 'shop_id', 'item_id',
                     'item_cnt_month', 'item_avg_cnt', 'shop_avg_cnt',
                     'cat_avg_cnt', 'avg_price', 'item_category_id']].copy()

    for lag in [1, 2, 3, 6, 12]:
        tmp = lag_source[lag_source['date_block_num'] == 34 - lag][
            ['shop_id', 'item_id', 'item_cnt_month', 'avg_price']
        ].rename(columns={
            'item_cnt_month': f'item_cnt_month_lag_{lag}',
            'avg_price': f'avg_price_lag_{lag}' if lag == 1 else 'avg_price_tmp'
        })
        if lag != 1:
            tmp = tmp.drop(columns=['avg_price_tmp'])
        test = test.merge(tmp, on=['shop_id', 'item_id'], how='left')

    item_data = lag_source[lag_source['date_block_num'].isin([33, 32, 31])].copy()
    item_data['lag'] = 34 - item_data['date_block_num']
    item_pivot = (item_data.pivot_table(index='item_id', columns='lag', values='item_avg_cnt')
                  .add_prefix('item_avg_cnt_lag_').reset_index())

    shop_data = lag_source[lag_source['date_block_num'].isin([33, 32, 31])].copy()
    shop_data['lag'] = 34 - shop_data['date_block_num']
    shop_pivot = (shop_data.pivot_table(index='shop_id', columns='lag', values='shop_avg_cnt')
                  .add_prefix('shop_avg_cnt_lag_').reset_index())

    test = test.merge(item_pivot, on='item_id', how='left')
    test = test.merge(shop_pivot, on='shop_id', how='left')

    ca_dict = lag_source[lag_source['date_block_num'] == 33].set_index(
        'item_category_id')['cat_avg_cnt'].to_dict()
    test['cat_avg_cnt_lag_1'] = test['item_category_id'].map(ca_dict)

    test['trend_1_2']  = test['item_cnt_month_lag_1'] - test['item_cnt_month_lag_2']
    test['trend_1_12'] = test['item_cnt_month_lag_1'] - test['item_cnt_month_lag_12']

    for win in [3, 6, 12]:
        test[f'item_cnt_month_rmean_{win}'] = test['item_cnt_month_lag_1']

    return test


def build_features(sales_train: pd.DataFrame,
                   items: pd.DataFrame,
                   test_raw: pd.DataFrame) -> tuple:
    """
    Главная функция. Принимает чистый sales_train, items, сырой test.
    Возвращает (df, test) готовые к обучению/инференсу.

    Использование в DAG:
        from sales_ds.features import build_features
        df, test = build_features(sales_train, items, test_raw)
    """
    print('1/6 Строим месячную сетку...')
    df = build_monthly_grid(sales_train)

    print('2/6 Лаговые фичи item_cnt_month...')
    df = add_lag_feature(df, [1, 2, 3, 6, 12], 'item_cnt_month')

    print('3/6 Групповые средние...')
    df = add_group_averages(df, items)

    print('4/6 Лаг avg_price + rolling means + trends...')
    df = add_lag_feature(df, [1], 'avg_price')
    df = add_rolling_means(df)
    df = add_trends(df)
    df = add_date_features(df)

    print('5/6 Фичи для тест сета...')
    test = build_test_features(test_raw, df, items)

    print('6/6 Дополнительные фичи (baseline)...')
    df, test = add_extra_features(df, test)

    print('✓ Feature engineering завершён')
    return df, test