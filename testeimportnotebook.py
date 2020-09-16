# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
import inflection
# import numpy as np


# %%
df_sales_raw = pd.read_csv('data/train.csv', low_memory = False) 
df_store_raw = pd.read_csv('data/store.csv', low_memory = False) 
df_raw = pd.merge(df_sales_raw, df_store_raw, how='left', on='Store')
# print(df_raw.columns)


# %%
cols_old = ['Store', 'DayOfWeek', 'Date', 'Sales', 'Customers', 'Open', 'Promo',
            'StateHoliday', 'SchoolHoliday', 'StoreType', 'Assortment',
            'CompetitionDistance', 'CompetitionOpenSinceMonth',
            'CompetitionOpenSinceYear', 'Promo2', 'Promo2SinceWeek',
            'Promo2SinceYear', 'PromoInterval']
snakecase = lambda x: inflection.underscore(x)
cols_new = list(map(snakecase, cols_old))
df_raw.columns = cols_new


# %%
print('Number of Rows:{}'.format(df_raw.shape[0]))
print('Number of Cols:{}'.format(df_raw.shape[1])) 


# %%



