#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 21:28:36 2019

@author: tauro
"""

idx = pd.date_range(df.date.min(), df.date.max())

aa = test[test['sku'] == 'SKU0933']


bb = aa.copy()

bb = bb.set_index('date')

bb.index = pd.DatetimeIndex(bb.index)

#bb = bb.drop(column=['date'])

bb = bb.reindex(idx)

bb = bb.resample('D').ffill().reset_index()



idx_ = pd.period_range(df.date.min(), df.date.max())

bb = bb.reindex(idx_, fill_value=0)

cc = bb.merge(aa, left_on=bb.index, right_on=aa['date'], how='left')



df1 = pd.DataFrame([[1, 3], [2, 4]], columns=['A', 'B'])

df2 = pd.DataFrame([[1, 5], [1, 6]], columns=['A', 'C'])


df1.merge(df2, how='left', on='A')







idx = pd.date_range(df.date.min(), df.date.max())

idx = pd.Index(idx)

bb.index = idx

