import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error

def prepro(target):
    # generating training data for predicting missing values
    # using other measurements to predict PM10
    aq = pd.read_csv('data/aq.csv')
    aq = aq.dropna(axis=0, how='any')
    x = np.array(aq[['PM2.5', 'NO2', 'CO', 'O3', 'SO2']])
    y = np.array(aq['PM10'])
    np.save('data/' + target + '_x', x)
    np.save('data/' + target + '_y', y)

def fill_missing():
    # fill out nan in aq data(except PM10)
    aq = pd.read_csv('data/aq.csv')
    for i in range(1, len(aq)):
        for j in ['PM2.5', 'NO2', 'CO', 'O3', 'SO2']:
            if pd.isnull(aq.iloc[i][j]):
                val = aq.iloc[i-1][j]
                aq.set_value(i, j, val)
    aq.to_csv('data/baq.csv')

def prepro_time():
    # generate training data for missing value prediction
    aq = pd.read_csv('data/pre_aq.csv')
    print(aq)
    exit()
    x, y = [], []
    for i in range(len(aq) - 2):
        s_x, s_y = [], []
        cur_row = aq.iloc[i]
        next_row = aq.iloc[i+1]
        nn_row = aq.iloc[i+2]
        if cur_row.isnull().values.any() or next_row.isnull().values.any() or nn_row.isnull().values.any():
            continue
        #if pd.isnull(cur_row['PM10']) or pd.isnull(next_row['PM10']) or pd.isnull(nn_row['PM10']):
        #    continue
        for m in ['PM2.5', 'PM10', 'NO2', 'CO', 'O3', 'SO2']:
            s_x.append(cur_row[m])
        for m in ['PM2.5', 'NO2', 'CO', 'O3', 'SO2']:
            s_x.append(next_row[m])
        for m in ['PM2.5', 'NO2', 'CO', 'O3', 'SO2']:
            s_x.append(nn_row[m])
        s_y.append(next_row['PM10'])
        x.append(s_x)
        y.append(s_y)
    np.save('data/PM10_x', np.array(x))
    np.save('data/PM10_y', np.array(y))
        
        
def train_model(target):
    # train model and fill out missing PM10 values
    x = np.load('data/' + target + '_x.npy')
    y = np.load('data/' + target + '_y.npy')
    size = int(x.shape[0] * 0.2)
    # 40000 samples for testing
    x_train, x_test = x[size:], x[:size]
    y_train, y_test = y[size:], y[:size]
    poly = PolynomialFeatures(degree=2)
    print(x_train.shape)
    x_train = poly.fit_transform(x_train)
    x_test = poly.fit_transform(x_test)
    #regr = RandomForestRegressor(max_depth=20, n_estimators=1200)
    regr = LinearRegression()
    regr.fit(x_train, y_train.ravel())
    pred = regr.predict(x_test)
    print(mean_absolute_error(y_test, pred))
    aq = pd.read_csv('data/baq.csv')
    for i in range(1, len(aq)-1):
        if pd.isnull(aq.iloc[i]['PM10']):
            s_x = []
            for m in ['PM2.5', 'PM10', 'NO2', 'CO', 'O3', 'SO2']:
                s_x.append(aq.iloc[i-1][m])
            for m in ['PM2.5', 'NO2', 'CO', 'O3' ,'SO2']:
                s_x.append(aq.iloc[i][m])
            for m in ['PM2.5', 'NO2', 'CO', 'O3', 'SO2']:
                s_x.append(aq.iloc[i+1][m])
            s_x = poly.fit_transform(np.array(s_x).reshape(1, -1))
            mv = regr.predict(s_x)
            aq.set_value(i, 'PM10', mv[0])
    aq.to_csv('data/pre_aq.csv')

def baseline():
    aq = pd.read_csv('data/baq.csv')
    count = 0
    tot = 0
    for i in range(len(aq)-1):
        cur_val = aq.iloc[i]['PM10']
        nex_val = aq.iloc[i+1]['PM10']
        if pd.isnull(cur_val) == True or pd.isnull(nex_val) == True:
            continue
        tot = tot + abs(cur_val - nex_val)
        count += 1
        if count == 10000:
            print(tot/count)
            return

fill_missing()
prepro_time()
train_model('PM10')
