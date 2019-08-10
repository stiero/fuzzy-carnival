#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 11:59:01 2019

@author: msr
"""

import pandas as pd
import glob

import matplotlib.pyplot as plt


import seaborn as sns

import datetime
import numpy as np

from helper import read_files, overlap

files = glob.glob("data/*.csv")

columns = ['sku', 'brand', 'cat', 'bill', 'store', 'date', 'type', 'mrp', 'price', 'qty']

df_dict = read_files(files, columns)
    


#A few checks for overlaps

overlap(df_dict['df1'], df_dict['df2'], 'sku')

overlap(df_dict['df1'], df_dict['df9'], 'sku')

overlap(df_dict['df1'], df_dict['df6'], 'brand')

overlap(df_dict['df4'], df_dict['df7'], 'bill')


df = pd.concat(df for df in df_dict.values())
    






#Reasonable to assume that separate df files can be merged into a bigger df

#df = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8, df9])











#Data cleaning and formatting

df['store'] = df['store'].apply(lambda x: x[-1])




# Checking for duplicated values

df.duplicated(keep='first').sum()

df = df.drop_duplicates(keep="first")



#Handling missing values

df.isna().sum()


df[df['brand'].isnull()]

df[df['cat'].isnull()]

df.loc[df['brand'].isnull(), 'brand'] = 'UnkBRAND'

df.loc[df['cat'].isnull(), 'cat'] = 'UnkCAT'




#Distribution of cols



skus = df.groupby('sku').size().sort_values(ascending=False)
skus.head(10)

cats = df.groupby('cat').size().sort_values(ascending=False)
cats

brands = df.groupby('brand').size().sort_values(ascending=False)
brands.head(10)

dates = df.groupby('date').size().sort_values(ascending=False)
dates.head(10)

types = df.groupby('type').size()

qtys = df.groupby('qty').size()

stores = df.groupby('store').size()




cat_plot = sns.countplot(df['cat'])
plt.xticks(rotation=45)

store_plot = sns.countplot(df['store'])
plt.xticks(rotation=45)

sale_type_plot = sns.countplot(df['type'])
plt.xticks(rotation=45)
    
qty_plot = sns.countplot(df['qty'])
plt.xticks(rotation=45)


sales = df[df['type'] == 'Sale']

returns = df[df['type'] == 'Return']

agg_sales = sales.groupby('store')['qty'].agg('sum')


#I noticed some have zero sales

zero_sales = df[df['qty'] == 0]

zero_sales_skus = zero_sales.sku.unique()


for sku in zero_sales_skus:
    print(set(df[df['sku'] == sku]['price']))


#They make no sense, so removing
df = df[df['qty'] != 0]



import scipy.stats as stats
stats.f_oneway(df['qty'][df['store'] == 'Store 1'], df['qty'][df['store'] == 'Store 2'], 
               df['qty'][df['store'] == 'Store 3'], df['qty'][df['store'] == 'Store 4'],
               df['qty'][df['store'] == 'Store 5'], df['qty'][df['store'] == 'Store 6'])















# Fixing the date

df['date'] = pd.to_datetime(df['date'])

df['day'] = pd.DatetimeIndex(df['date']).day

df['month'] = pd.DatetimeIndex(df['date']).month_name()

df['dayofweek'] = pd.DatetimeIndex(df['date']).day_name()


weekend_days = ['Saturday', 'Sunday']

df['weekend'] = np.where(df['dayofweek'].isin(weekend_days), 1, 0)



# See if price is greater than MRP (overcharging)

overcharged = df[df['price'] > df['mrp'] ]

#Overcharging does happen on 1665 transactions. Some of these SKUs have MRP = 0.
#
#One possible explanation for this is that these SKUs were intended to be discounted, but were sold at a higher price, mistakenly or otherwise
#
#For the sake of simplicity, we rectify it by forcing the price and mrp to be equal. 



df['mrp'] = np.where(df.price > df.mrp, df.price, df.mrp)



zero_mrp = df[df['mrp'] == 0]

zero_mrp_skus = zero_mrp.sku.unique().tolist()

#Are these SKUs given out for zero mrp every time? 

zero_mrp_skus_avg_mrps = df[(df['sku'].isin(zero_mrp_skus)) & (df['mrp'] != 0)].groupby('sku')['mrp'].agg(lambda x : np.mean(x))






for sku, avg in zero_mrp_skus_avg_mrps.items():
    df.loc[(df['sku'] == sku) & (df['mrp'] == 0), ['mrp']] = avg

df[(df['sku'].isin(zero_mrp_skus)) & (df['mrp'] == 0)]['mrp']


df['total'] = df['price'] * df['qty']

df['discount'] = df['mrp'] - df['price']

# We see negative discounts and discounts above 100% (which means money has been given back)

df['perc_discount'] = ((df['mrp'] - df['price']) / df['mrp']) * 100




#df['date'] = df['date'].astype('datetime64')


#df1['Sale Date'].nunique()


#df1['Sale Date'] = pd.to_datetime(df1['Sale Date'])


df_sales_days = df.groupby('date')['qty'].sum()

df_date_sorted = df.sort_values(by='date')

dates = df['date'].unique().tolist()


#dates = list(df1['Sale Date'])

sns.lineplot(data = df_sales_days)
