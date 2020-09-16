#%% 0.0 Imports
import pandas as pd
import inflection
import math
# import numpy as np

#%% 0.1 Helper Functions


#%% 0.1 Loading data

df_sales_raw = pd.read_csv('data/train.csv', low_memory = False) 
df_store_raw = pd.read_csv('data/store.csv', low_memory = False) 
df_raw = pd.merge(df_sales_raw, df_store_raw, how='left', on='Store')
# print(df_raw.columns)

#%% 1.0 Data Description
df1 = df_raw.copy()

#%% 1.1 Rename columns
cols_old = ['Store', 'DayOfWeek', 'Date', 'Sales', 'Customers', 'Open', 'Promo',
            'StateHoliday', 'SchoolHoliday', 'StoreType', 'Assortment',
            'CompetitionDistance', 'CompetitionOpenSinceMonth',
            'CompetitionOpenSinceYear', 'Promo2', 'Promo2SinceWeek',
            'Promo2SinceYear', 'PromoInterval']
snakecase = lambda x: inflection.underscore(x)
cols_new = list(map(snakecase, cols_old))
df1.columns = cols_new

#%% 1.2 Data Dimension
print('Number of Rows:{}'.format(df1.shape[0]))
print('Number of Cols:{}'.format(df1.shape[1])) 

#%% 1.3 Data Types
df1['date'] = pd.to_datetime(df1['date'])
df1.dtypes


#%% 1.4 Check NA
df1.isna().sum()
# df1['competition_distance'].max()

#%% 1.5 Fillout NA

#%% 1.5 Fillout NA # competition_distance              2642
df1['competition_distance'] = df1['competition_distance'].apply( lambda x: 200000.0 if math.isnan(x) else x )


#%% 1.5 Fillout NA # competition_open_since_month    323348
df1['competition_open_since_month'] =  df1.apply( lambda x: df1['date'].dt.month if math.isnan(x['competition_open_since_month']) else x['competition_open_since_month'], axis=1 )

#%% 1.5 Fillout NA # competition_open_since_year     323348
df1['competition_open_since_year'] =  df1.apply( lambda x: df1['date'].dt.year if math.isnan(x['competition_open_since_year']) else x['competition_open_since_year'], axis=1 )

#%% 1.5 Fillout NA # promo2_since_week               508031
df1['promo2_since_week'] =  df1.apply( lambda x: df1['date'].dt.week if math.isnan(x['promo2_since_week']) else x['promo2_since_week'], axis=1 )

#%% 1.5 Fillout NA # promo2_since_year               508031
df1['promo2_since_year'] =  df1.apply( lambda x: df1['date'].dt.year if math.isnan(x['promo2_since_year']) else x['promo2_since_year'], axis=1 )




#%%
# df1["date"].month

month_map = {1: "Jan", 2: "Fev", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun", 7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dez" }
df1["promo_interval"].fillna(0, inplace=True)
df1["month_map"] = df1["date"].dt.month.map (month_map)
#%%
df1.sample(5)["date"].dt.year
df1.sample(5)["date"]
