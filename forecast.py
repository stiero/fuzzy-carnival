#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 14:10:58 2019

@author: msr
"""

import pandas as pd
import glob

import matplotlib.pyplot as plt


import seaborn as sns

import datetime
import numpy as np

from helper import read_files

files = glob.glob("data/*.csv")

columns = ['sku', 'brand', 'cat', 'bill', 'store', 'date', 'type', 'mrp', 'price', 'qty']

df_dict = read_files(files, columns)

df = pd.concat(df for df in df_dict.values())
df = df.drop(columns=['source'])


df['store'] = df['store'].apply(lambda x: x[-1])





df = df[df['store'] == '5']

df = df.sort_values(by='date').reset_index(drop=True)

df.loc[df['brand'].isnull(), 'brand'] = 'UnkBRAND'

df.loc[df['cat'].isnull(), 'cat'] = 'UnkCAT'






df['date'] = pd.to_datetime(df['date'])

df['day'] = pd.DatetimeIndex(df['date']).day

df['month'] = pd.DatetimeIndex(df['date']).month_name()

df['month_year'] = df.date.dt.to_period('M')

df['dayofweek'] = pd.DatetimeIndex(df['date']).day_name()

weekend_days = ['Saturday', 'Sunday']

df['weekend'] = np.where(df['dayofweek'].isin(weekend_days), 1, 0)

df['weekno'] = df.date.dt.week

df = df[df['qty'] != 0]




df['mrp'] = np.where(df.price > df.mrp, df.price, df.mrp)

df =  df[df['mrp'] != 0]

df = df[df['type'] == 'Sale']



df['total'] = df['price'] * df['qty']

df['discount'] = df['mrp'] - df['price']

df['perc_discount'] = ((df['mrp'] - df['price']) / df['mrp']) * 100

df['perc_discount'] = np.where(df['perc_discount'] == -np.inf, 0, df['perc_discount'])




from helper import week_number_from_start

df['weeks_from_start'] = 0

df = week_number_from_start(df)


df = df.drop_duplicates(keep="first")



qty_by_date = df.groupby(['date', 'sku'])['qty'].sum().reset_index()

test = pd.merge(qty_by_date, df.drop_duplicates(subset=['date', 'sku']), 
                on=['date', 'sku'], how='inner')



df.dtypes



#Creating more features

#Price stratification


#More than one item in bill

bill_count = df.groupby('bill').count().iloc[:,1].to_frame()


for index, value in bill_count.iterrows():
    df.loc[df['bill'] == index, 'bill_count'] = int(value[0])
    
df = df.drop(columns=['bill'])


# Rolling sales for 30 days

rolling = df.groupby('sku')['qty'].rolling(30, min_periods=1, on=list(df.date)).sum()


rolling = df.reset_index(drop=True).set_index('date').groupby('sku')['qty'].rolling(window=30, min_periods=1).sum()






idx = pd.date_range(df.date.min(), df.date.max())

grouped = test.groupby('sku')

final = pd.DataFrame()

for group in grouped.groups:
    frame = grouped.get_group(group)
    
    frame = frame.set_index('date')
    
    frame = frame.reindex(idx)
    
    frame = frame.reset_index()
    
    frame = frame.rename(columns={"index": "date","qty_x": "qty"})

    frame['qty'] = frame['qty'].fillna(0)

    frame = frame.drop(columns=["qty_y", "bill"])
    
    frame['roll_30'] = frame['qty'].rolling(30, min_periods=1).sum()
    
    frame['roll_60'] = frame['qty'].rolling(60, min_periods=1).sum()
    
    frame['roll_90'] = frame['qty'].rolling(90, min_periods=1).sum()
    
    frame = frame.dropna()
    
    final = pd.concat([final, frame], axis=0)


final = final.sort_values('date').reset_index().drop(columns=['index'])



    
#    frame = frame.reset_index(drop=True).set_index('date')
#    frame['roll_30'] = frame['qty_x'].rolling(30, min_periods=1).sum()
#    coll.append(frame)

df['roll_30'] = df.qty.rolling(30, min_periods=1).sum()


rolling = df.reset_index(drop=True).set_index('date').groupby('sku')['qty'].rolling(window=30, min_periods=1, 
                        o).sum()


