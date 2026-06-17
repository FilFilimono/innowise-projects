import json
import os
import pandas as pd
import numpy as np



def check_missing_data(df: pd.DataFrame) -> dict:

    missing_data = df.isnull().sum()
    missing_data_pct = (missing_data / len(df)) * 100

    column_types = {}
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            column_types[col] = 'numeric'
        else:
            column_types[col] = 'categorical'

    return {
        'has_missing': bool(missing_data.sum() > 0),
        'total_missing': int(missing_data.sum()),
        'missing_data': missing_data[missing_data > 0].to_dict(),
        'missing_pct': missing_data_pct.to_dict(),
        'columns': df.columns[df.isnull().any()].tolist(),
        'column_types': column_types,
        'suggested_action': ''
    }


def check_outliers_top_k(df: pd.DataFrame, column: str, k: int = 2) -> dict:

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    upper_iqr = Q3 + 1.5 * IQR
    lower_iqr = Q1 - 1.5 * IQR

    high_outliers = df[df[column] > upper_iqr].nlargest(k, column)
    low_outliers = df[df[column] < lower_iqr].nsmallest(k, column)
    top_outliers = pd.concat([high_outliers, low_outliers])

    if len(top_outliers) > k:
        top_outliers['deviation'] = abs(top_outliers[column] - df[column].median())
        top_outliers = top_outliers.nlargest(k, 'deviation').drop(columns=['deviation'])

    if len(top_outliers) == 0:
        return {
            'has_outliers': False,
            'k_outliers': 0,
            'total_outliers': 0,
            'values_to_remove': [],
            'suggested_action': ''
        }

    values_to_remove = top_outliers[column].tolist()
    return {
        'has_outliers': True,
        'k_outliers': len(values_to_remove),
        'total_outliers': int(len(df[(df[column] > upper_iqr) | (df[column] < lower_iqr)])),
        'values_to_remove': values_to_remove,
        'upper_threshold': float(top_outliers[column].max()),
        'lower_threshold': float(top_outliers[column].min()),
        'k': k,
        'suggested_action': ''
    }


def check_duplicates(df: pd.DataFrame) -> dict:
   
    total_duplicates = int(df.duplicated().sum())
    return {
        'has_duplicates': total_duplicates > 0,
        'total_duplicates': total_duplicates,
        'suggested_action': ''
    }


def check_negative_values(df: pd.DataFrame, column: str) -> dict:
   
    total_invalid = int(len(df[df[column] < 0]))
    return {
        'has_negative': total_invalid > 0,
        'total_negative': total_invalid,
        'suggested_action': ''
    }




def make_summary(df_dict: dict) -> dict:

    summary = {}
    positive_cols = ['item_price', 'item_cnt_day', 'date_block_num']

    for name, df in df_dict.items():
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        summary[name] = {}

        summary[name]['missing'] = check_missing_data(df)

        summary[name]['duplicates'] = check_duplicates(df)

        summary[name]['outliers'] = {}
        for col in numeric_cols:
            summary[name]['outliers'][col] = check_outliers_top_k(df, col, k=2)

        summary[name]['negative_values'] = {}
        for col in numeric_cols:
            if col in positive_cols or any(x in col for x in ['price', 'cnt', 'value']):
                summary[name]['negative_values'][col] = check_negative_values(df, col)

        summary[name]['dtypes'] = df.dtypes.astype(str).to_dict()
        summary[name]['shape'] = {'rows': len(df), 'columns': len(df.columns)}

    return summary




def etl_layer(raw_data_path: str, clean_data_path: str, json_path: str) -> None:
    with open(json_path, 'r') as f:
        issues_summary = json.load(f)

    raw_data = {
        'sales_train': pd.read_csv(f'{raw_data_path}/sales_train.csv'),
        'shops':       pd.read_csv(f'{raw_data_path}/shops.csv'),
        'items':       pd.read_csv(f'{raw_data_path}/items.csv'),
        'item_categories': pd.read_csv(f'{raw_data_path}/item_categories.csv'),
    }

    clean_data = {}

    for df_name, df in raw_data.items():
        if df_name not in issues_summary:
            clean_data[df_name] = df
            print(f'[SKIP] {df_name} не найден в summary')
            continue

        print(f'\n[{df_name}]')
        info = issues_summary[df_name]

        dup = info.get('duplicates', {})
        if dup.get('suggested_action') == 'remove':
            before = len(df)
            df = df.drop_duplicates()
            print(f'  Дубликаты удалены: {before - len(df)} строк')

        for col, out in info.get('outliers', {}).items():
            if out.get('suggested_action') == 'remove' and col in df.columns:
                before = len(df)
                for val in out['values_to_remove']:
                    df = df[df[col] != val]
                print(f'  Выбросы в {col} удалены: {before - len(df)} строк')

        for col, neg in info.get('negative_values', {}).items():
            action = neg.get('suggested_action')
            if action == 'set_to_zero' and col in df.columns:
                count = (df[col] < 0).sum()
                df.loc[df[col] < 0, col] = 0
                print(f'  {col}: {count} отрицательных → 0')
            elif action == 'remove_rows' and col in df.columns:
                before = len(df)
                df = df[df[col] >= 0]
                print(f'  {col}: удалено {before - len(df)} строк с отрицательными')

        miss = info.get('missing', {})
        miss_action = miss.get('suggested_action')
        if miss_action == 'remove_rows':
            before = len(df)
            df = df.dropna()
            print(f'  Пропуски: удалено {before - len(df)} строк')
        elif miss_action == 'fill_median':
            for col in miss.get('columns', []):
                if col in df.columns and df[col].dtype in ['int64', 'float64']:
                    median = df[col].median()
                    df[col] = df[col].fillna(median)
                    print(f'  {col}: пропуски заполнены медианой ({median:.2f})')
        elif miss_action == 'fill_mean':
            for col in miss.get('columns', []):
                if col in df.columns and df[col].dtype in ['int64', 'float64']:
                    mean = df[col].mean()
                    df[col] = df[col].fillna(mean)
                    print(f'  {col}: пропуски заполнены средним ({mean:.2f})')

        clean_data[df_name] = df

    os.makedirs(clean_data_path, exist_ok=True)
    for df_name, df in clean_data.items():
        out_path = f'{clean_data_path}/{df_name}.csv'
        df.to_csv(out_path, index=False)
        print(f'  Сохранено: {out_path}')

    print('\n✓ ETL завершён')


def validate_schema(df: pd.DataFrame) -> None:
    required = ['date_block_num', 'shop_id', 'item_id', 'item_price', 'item_cnt_day']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f'Отсутствуют колонки: {missing}')
    print('✓ Схема данных валидна')