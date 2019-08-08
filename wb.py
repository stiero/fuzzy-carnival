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

files = glob.glob("*.csv")

columns = ['sku', 'brand', 'cat', 'bill', 'store', 'date', 'type', 'mrp', 'price', 'qty']

#df_dict = {}
#
#for file in files:
#    df = pd.read_csv(file, names=columns, header=None)
#    df_dict[file] = df
#    
#
#
#df = pd.concat((pd.read_csv(f) for f in files))    
#
#df.columns = columns


df1 = pd.read_csv("file1.csv", names = columns, header=0)
df1['source'] = 1

df2 = pd.read_csv("file2.csv", names = columns, header=0)
df2['source'] = 2

df3 = pd.read_csv("file3.csv", names = columns, header=0)
df3['source'] = 3

df4 = pd.read_csv("file4.csv", names = columns, header=0)
df4['source'] = 4

df5 = pd.read_csv("file5.csv", names = columns, header=0)
df5['source'] = 5

df6 = pd.read_csv("file6.csv", names = columns, header=0)
df6['source'] = 6

df7 = pd.read_csv("file7.csv", names = columns, header=0)
df7['source'] = 7

df8 = pd.read_csv("file8.csv", names = columns, header=0)
df8['source'] = 8

df9 = pd.read_csv("file9.csv", names = columns, header=0)
df9['source'] = 9


#A few checks for overlaps

overlap(df1, df2, 'sku')

overlap(df1, df9, 'sku')

overlap(df1, df6, 'brand')


#Reasonable to assume that separate df files can be merged into a bigger df

df = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8, df9])

df['store'] = df['store'].apply(lambda x: x[-1])







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

#df['date'] = df['date'].astype('datetime64')


#df1['Sale Date'].nunique()


#df1['Sale Date'] = pd.to_datetime(df1['Sale Date'])


df = df.sort_values(by='date')

dates = df['date'].unique().tolist()


dates = list(df1['Sale Date'])

sns.tsplot(data = df1['Sales Qty'], time = df1['Sale Date'])
