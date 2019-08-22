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

from sklearn.metrics import mean_squared_error

from tqdm import tqdm
tqdm.pandas(desc="progress-bar")

import statistics


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

#df['month_year'] = df.date.dt.to_period('M')

df['dayofweek'] = pd.DatetimeIndex(df['date']).day_name()

weekend_days = ['Saturday', 'Sunday']

df['weekend'] = np.where(df['dayofweek'].isin(weekend_days), 1, 0)

df['weekno'] = df.date.dt.week

df = df[df['qty'] != 0]




df['mrp'] = np.where(df.price > df.mrp, df.price, df.mrp)

df =  df[df['mrp'] != 0]

#df = df[df['type'] == 'Sale']



df['total'] = df['price'] * df['qty']

df['discount'] = df['mrp'] - df['price']

df['perc_discount'] = ((df['mrp'] - df['price']) / df['mrp']) * 100

df['perc_discount'] = np.where(df['perc_discount'] == -np.inf, 0, df['perc_discount'])




#from helper import week_number_from_start

#df['weeks_from_start'] = 0

#df = week_number_from_start(df)


df = df.drop_duplicates(keep="first")



qty_by_date = df.groupby(['date', 'sku'])['qty'].sum().reset_index()

df = pd.merge(qty_by_date, df.drop_duplicates(subset=['date', 'sku']), 
                on=['date', 'sku'], how='inner')


df = df.rename(columns={"qty_x": "qty"})

df = df.drop(columns=["qty_y"])



df.dtypes



#Creating more features

#Price stratification


#More than one item in bill

bill_count = df.groupby('bill').count().iloc[:,1].to_frame()


for index, value in tqdm(bill_count.iterrows()):
    df.loc[df['bill'] == index, 'bill_item_count'] = int(value[0])
    
df = df.drop(columns=['bill'])


# Rolling sales for 30 days

#rolling = df.groupby('sku')['qty'].rolling(30, min_periods=1, on=list(df.date)).sum()
#
#
#rolling = df.reset_index(drop=True).set_index('date').groupby('sku')['qty'].rolling(window=30, min_periods=1).sum()





idx = pd.date_range(df.date.min(), df.date.max())

grouped = df.groupby('sku')

final = pd.DataFrame()

for group in tqdm(grouped.groups):
    frame = grouped.get_group(group)
    
    frame_cat = frame['cat'].unique()[0]
    
    frame_cat = frame['cat'].unique()[0]
    
    frame_sku = frame['sku'].unique()[0]
    
    frame_brand = frame['brand'].unique()[0]




    
    frame = frame.set_index('date')
    
    frame = frame.reindex(idx)
    
    frame = frame.reset_index()
    
    frame = frame.rename(columns={"index": "date"})
    
    
    

    frame['qty'] = frame['qty'].fillna(0)
    
    frame['sku'] = frame['sku'].fillna(frame_sku)
    
    frame['cat'] = frame['cat'].fillna(frame_cat)
    
    frame['brand'] = frame['brand'].fillna(frame_brand)
    
    frame['mrp'] = frame['mrp'].fillna(method='ffill')
    
    frame['price'] = frame['price'].fillna(method='ffill')
    
    frame['bill_item_count'] = frame['bill_item_count'].fillna(method='ffill')
    
    frame['roll_30'] = frame['qty'].rolling(30, min_periods=1).sum()
    
    frame['roll_60'] = frame['qty'].rolling(60, min_periods=1).sum()
    
    frame['roll_90'] = frame['qty'].rolling(90, min_periods=1).sum()
    
    frame = frame.dropna()
    
    final = pd.concat([final, frame], axis=0)


final = final.sort_values('date').reset_index().drop(columns=['index'])

#final.to_csv("final_all_dates.csv",  index=False)

#final = pd.read_csv("final_all_dates.csv")





skus = final.sku.unique()

dates = final.date.unique()










cat_cols = ['sku', 'brand', 'cat', 'store', 'type', 'day', 'month', 'dayofweek', 
            'weekend', 'weekno']

#test = final.copy()

for col in tqdm(cat_cols):
    final = pd.concat([final, pd.get_dummies(final[col], drop_first=True,
                                             prefix=col)], axis=1)
    del final[col]
    





#skus = final.sku.nunique()
#
#from sklearn.feature_extraction import FeatureHasher
#
#hasher = FeatureHasher(n_features=skus, input_type='string')
#
#aa = hasher.transform(test['sku'])


    
#    frame = frame.reset_index(drop=True).set_index('date')
#    frame['roll_30'] = frame['qty_x'].rolling(30, min_periods=1).sum()
#    coll.append(frame)

#df['roll_30'] = df.qty.rolling(30, min_periods=1).sum()


#rolling = df.reset_index(drop=True).set_index('date').groupby('sku')['qty'].rolling(window=30, min_periods=1, 
#                        o).sum()



cutoff_date = '2018-02-01'

X_train = final[final['date'] < cutoff_date]

X_test = final[final['date'] >= cutoff_date]

#del final['date']
del X_train['date']
del X_test['date']

y_train = X_train['qty']
y_test = X_test['qty']

del X_train['qty']
del X_test['qty']







from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

scale_cols = ['mrp', 'total', 'price', 'discount', 'perc_discount']

for col in tqdm(scale_cols):
    sc.fit(np.array(X_train[col]).reshape(-1, 1))
    
    X_train[col] = sc.transform(np.array(X_train[col]).reshape(-1,1)).astype(np.float16)
    
    X_test[col] = sc.transform(np.array(X_test[col]).reshape(-1,1)).astype(np.float16)
    
    





X_train = X_train.to_numpy()
y_train = y_train.to_numpy(dtype="int")


X_test = X_test.to_numpy()
y_test = y_test.to_numpy(dtype="int")














from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr_model = lr.fit(X_train, y_train)

lr_preds = lr_model.predict(X_test)


lr_score = lr_model.score(X_test, y_test)

mean_squared_error(y_test, lr_preds)










import statsmodels.api as sm

import statsmodels.formula.api as smf

from statsmodels.stats.outliers_influence import variance_inflation_factor


lr = sm.OLS(y_train, X_train)

lr_model = lr.fit()

lr_preds = lr_model.predict(X_test)

lr_preds.summary()


# For each X, calculate VIF and save in dataframe
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X_train, i) for i in range(X_train.shape[1])]
vif["features"] = X_train.columns







#
#
#from sklearn.preprocessing import PolynomialFeatures
#
#
#poly = PolynomialFeatures(degree=2)
#
#X_train = poly.fit_transform(X_train)
#    
#y_train = poly.fit_transform(y_train)


from helper import rmse


from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(max_depth=20, random_state=0,
                           n_estimators=2000, oob_score=True,
                           n_jobs=-1, verbose=0,
                           max_features='auto')

rf_model = rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

rf_score = rf.score(X_test, y_test)

rmse(rf_pred, y_test), rf_model.oob_score_, rf_score

mean_squared_error(y_test, rf_pred)





#xgbMatrix_train = xgb.DMatrix(data=X_train, label=y_train)
#
#xgbMatrix_test = xgb.DMatrix(data=X_test)
#
#
#params = {'max_depth': 2, 'eta': 0.5, 'silent': 1, 'objective': 'reg:squarederror',
#          'nthread': 4, 'colsample_bytree': 0.7, 'subsample': 0.5, 
#          'scale_pos_weight': 1, 'gamma': 5, 'learning_rate': 0.02,
#          'num_boost_round': 100}
#
#
#xgb_model = xgb.train(params, xgbMatrix_train)
#
#
#xgb_pred = xgb_model.predict(xgbMatrix_test)
#
#rmse(xgb_pred, y_test)

import xgboost as xgb

xg_reg = xgb.XGBRegressor(max_depth= 5, eta= 0.5, silent= 1, 
                          objective= 'reg:squarederror', nthread= 4, 
                          colsample_bytree= 0.7, subsample= 0.5, 
                          scale_pos_weight= 1, gamma= 5, learning_rate= 0.05, 
                          num_boost_round= 100)


xgb_model = xg_reg.fit(X_train, y_train)

xgb_pred = xgb_model.predict(X_test)

mean_squared_error(y_test, xgb_pred)



xg_reg = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.4,
                max_depth = 5, alpha = 10, n_estimators = 1000)

xgb_model = xg_reg.fit(X_train, y_train)

xgb_pred = xgb_model.predict(X_test)


rmse(xgb_pred, y_test)




skus = final.sku.unique()

#def train_generator(final, skus):
#    final_sku = final


ndim = X_train.shape[-1]


from keras.models import Sequential
from keras.layers import LSTM, GRU, Flatten, Dense, Dropout
from keras.optimizers import RMSprop, Adam
from keras.callbacks import EarlyStopping
from keras.utils  import to_categorical



y_binary_train = to_categorical(y_train-1, num_classes=12)

y_binary_test = to_categorical(y_test-1, num_classes=12)

callbacks_list = [EarlyStopping(monitor='loss',
                                patience=10)]

model = Sequential()
model.add(Dense(32, input_dim=ndim))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(12, activation="softmax"))

model.compile(optimizer=Adam(), loss='categorical_crossentropy',
              metrics = ['accuracy'])

history = model.fit(X_train, y_binary_train,
                    epochs=300,
                    batch_size=16,
                    callbacks=callbacks_list)


model_pred = model.predict(X_test)

mappings = {}

for actual, encoded in zip(y_train, y_binary_train):
    mappings[str(np.argmax(encoded))] = actual

for actual, pred in zip(y_test, model_pred):
    print(actual, mappings[str(np.argmax(pred))])



















