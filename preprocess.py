import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler


class Cardiopreprocess:
    def __init__(self):
        self.ap_hi_median = 120.0
        self.ap_lo_median = 80.0
        self.mean_height = 164.362
        self.mean_weight = 74.203

    def preprocessdata(self, data):
        #Train data don't have missing values. Filling missing values in prediction data if any.
        data.fillna(data.mean(), inplace=True)

        data['bmi'] = data['weight'] / (data['height'] / 100) ** 2
        data['Cho_glu_active'] = data['cholesterol'] + data['gluc'] + data['active']
        data.drop(['cholesterol', 'gluc', 'active'], axis=1, inplace=True)
        data['age'] = (data['age'] / 365).round().astype('int')
        data['gender'] = data['gender'].map({1: 0, 2: 1})
        data.loc[(data['ap_hi'] >= 250), 'ap_hi'] = self.ap_hi_median
        data.loc[(data['ap_hi'] <0), 'ap_hi'] = self.ap_hi_median
        data.loc[(data['ap_lo'] >= 200), 'ap_lo'] = self.ap_lo_median
        data.loc[(data['ap_lo']  <0), 'ap_lo'] = self.ap_lo_median
        data.loc[(data['ap_lo'] > data['ap_hi']), 'ap_lo'] = self.ap_lo_median
        data.loc[((data['height'] > data['height'].quantile(0.975)) | (
                data['height'] < data['height'].quantile(0.025))), 'height'] = self.mean_height
        data.loc[((data['weight'] > data['weight'].quantile(0.975)) | (
                data['weight'] < data['weight'].quantile(0.025))), 'weight'] = self.mean_weight
        scaler = joblib.load('minmaxscalar.pckl')
        data[data.columns] = scaler.transform(data[data.columns])
        return data
