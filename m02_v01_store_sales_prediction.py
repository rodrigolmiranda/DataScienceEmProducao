# %% 0.0 Imports
import pandas as pd
import inflection
import math
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
from IPython.core.display import HTML
from IPython.display import Image
# %% 0.1 Helper Functions


# %% 0.1 Loading data

df_sales_raw = pd.read_csv('data/train.csv', low_memory=False)
df_store_raw = pd.read_csv('data/store.csv', low_memory=False)
df_raw = pd.merge(df_sales_raw, df_store_raw, how='left', on='Store')
# print(df_raw.columns)

# %% 1.0 STEP 01 - Data Description
df1 = df_raw.copy()

# %% 1.1 Rename columns
cols_old = ['Store', 'DayOfWeek', 'Date', 'Sales', 'Customers', 'Open', 'Promo',
            'StateHoliday', 'SchoolHoliday', 'StoreType', 'Assortment',
            'CompetitionDistance', 'CompetitionOpenSinceMonth',
            'CompetitionOpenSinceYear', 'Promo2', 'Promo2SinceWeek',
            'Promo2SinceYear', 'PromoInterval']


def snakecase(x): return inflection.underscore(x)


cols_new = list(map(snakecase, cols_old))
df1.columns = cols_new

# %% 1.2 Data Dimension
print('Number of Rows:{}'.format(df1.shape[0]))
print('Number of Cols:{}'.format(df1.shape[1]))

# %% 1.3 Data Types
df1['date'] = pd.to_datetime(df1['date'])
df1.dtypes


# %% 1.4 Check NA
df1.isna().sum()
# df1['competition_distance'].max()

# %% 1.5 Fillout NA

# 1.5 Fillout NA # competition_distance              2642
df1['competition_distance'] = df1['competition_distance'].apply(
    lambda x: 200000.0 if math.isnan(x) else x)

# 1.5 Fillout NA # competition_open_since_month    323348
df1['competition_open_since_month'] = df1.apply(lambda x: x['date'].month if math.isnan(
    x['competition_open_since_month']) else x['competition_open_since_month'], axis=1)

# 1.5 Fillout NA # competition_open_since_year     323348
df1['competition_open_since_year'] = df1.apply(lambda x: x['date'].year if math.isnan(
    x['competition_open_since_year']) else x['competition_open_since_year'], axis=1)

# 1.5 Fillout NA # promo2_since_week               508031
df1['promo2_since_week'] = df1.apply(lambda x: x['date'].week if math.isnan(
    x['promo2_since_week']) else x['promo2_since_week'], axis=1)

# 1.5 Fillout NA # promo2_since_year               508031
df1['promo2_since_year'] = df1.apply(lambda x: x['date'].year if math.isnan(
    x['promo2_since_year']) else x['promo2_since_year'], axis=1)

# 1.5 create columns promo interval, promo_map, ispromo
month_map = {1: "Jan", 2: "Fev", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
             7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dez"}
df1["promo_interval"].fillna(0, inplace=True)
df1["month_map"] = df1["date"].dt.month.map(month_map)
df1["ispromo"] = df1[['promo_interval', 'month_map']].apply(
    lambda x: 0 if x["promo_interval"] == 0 else 1 if x["month_map"] in x["promo_interval"].split(',') else 0, axis=1)

# df1.sample(5).T


# %% 1.6 Change data types
df1.dtypes
df1["competition_open_since_month"] = df1["competition_open_since_month"].astype(
    int)
df1["competition_open_since_year"] = df1["competition_open_since_year"].astype(
    int)
df1["promo2_since_week"] = df1["promo2_since_week"].astype(int)
df1["promo2_since_year"] = df1["promo2_since_year"].astype(int)

# %% 1.7 Descriptive Statistical
num_attributes = df1.select_dtypes(include=["int64", "float64"])
cat_attributes = df1.select_dtypes(
    exclude=["int64", "float64", "datetime64[ns]"])

# %% 1.71 Numerical Attributes
# Central Tendency = mean, median
ct1 = pd.DataFrame(num_attributes.apply(np.mean)).T
ct2 = pd.DataFrame(num_attributes.apply(np.median)).T

# Dispersion - std, min, max, range, skew, kurtosis

d1 = pd.DataFrame(num_attributes.apply(np.std)).T
d2 = pd.DataFrame(num_attributes.apply(min)).T
d3 = pd.DataFrame(num_attributes.apply(max)).T
d4 = pd.DataFrame(num_attributes.apply(lambda x: x.max() - x.min())).T
d5 = pd.DataFrame(num_attributes.apply(lambda x: x.skew)).T
d6 = pd.DataFrame(num_attributes.apply(lambda x: x.kurtosis)).T

m = pd.concat([d2, d3, d4, ct1, ct2, d1, d5, d6]).T.reset_index()
m.columns = ['attributes', 'min', 'max', 'range',
             'mean', 'median', 'std', 'skew', 'kurtosis']
m

# %%
# df1.dtypes
# cat_attrubutes.sample(2)
# sns.distplot (df1["sales"])
# sns.distplot(df1["competition_open_since_year"])

# a = [1, 1, 1, 2, 2, 2, 5, 5, 5, 10]
# np.mean(a)
# np.median(a)


# %% 1.72 Categorical Attributes
# cat_attributes.apply( lambda x: x.unique().shape[0])
aux1 = df1[(df1['state_holiday'] != '0') & (df1['sales'] > 0)]

plt.subplot(1, 3, 1)
sns.boxplot(x="state_holiday", y="sales", data=aux1)

plt.subplot(1, 3, 2)
sns.boxplot(x="store_type", y="sales", data=aux1)

plt.subplot(1, 3, 3)
sns.boxplot(x="assortment", y="sales", data=aux1)

# %% 2.0 Step 02 - FEATURE ENGINEERING
Image('IMG/mindmaphipotesis.png')

# %% 2.1 Hipotesis creation

# %% 2.1.1 Hipotesis creation
