# import modules
import pandas as pd
import numpy as np
import os
import joblib
from warnings import filterwarnings
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

filterwarnings('ignore')

# read data
data = pd.read_csv('cardio_train.csv', delimiter=';')

data.drop('id', axis=1, inplace=True)

# map gender
data['gender'] = data['gender'].map({1: 0, 2: 1})

# BMI
data['bmi'] = data['weight'] / (data['height'] / 100) ** 2

# convert age to years
data['age'] = (data['age'] / 365).round().astype('int')

# Combine Choloestrol, glucose and physical activate columns
data['Cho_glu_active'] = data['cholesterol'] + data['gluc'] + data['active']
data.drop(['cholesterol', 'gluc', 'active'], axis=1, inplace=True)

# outlier treatment
ap_hi_median = data['ap_hi'].median()
ap_lo_median = data['ap_lo'].median()
mean_height = data['height'].mean()
mean_weight = data['weight'].mean()

data.loc[(data['ap_hi'] >= 250), 'ap_hi'] = ap_hi_median
data.loc[(data['ap_hi'] < 0), 'ap_hi'] = ap_hi_median
data.loc[(data['ap_lo'] >= 200), 'ap_lo'] = ap_lo_median
data.loc[(data['ap_lo'] < 0), 'ap_lo'] = ap_lo_median
data.loc[(data['ap_lo'] > data['ap_hi']), 'ap_lo'] = ap_lo_median

data.loc[((data['height'] > data['height'].quantile(0.975)) | (
            data['height'] < data['height'].quantile(0.025))), 'height'] = mean_height
data.loc[((data['weight'] > data['weight'].quantile(0.975)) | (
            data['weight'] < data['weight'].quantile(0.025))), 'weight'] = mean_weight

# Train test split
data_copy = data.copy(deep=True)
X = data_copy.drop('cardio', axis=1)
y = data_copy['cardio']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# normalize the data
from sklearn.preprocessing import MinMaxScaler

minmaxscalaer = MinMaxScaler()
X_train[X_train.columns] = minmaxscalaer.fit_transform(X_train[X_train.columns])
X_test[X_test.columns] = minmaxscalaer.transform(X_test[X_test.columns])

# fit model no training data
model = XGBClassifier()
model.fit(X_train, y_train)

# Model save
modelfile = 'model.pckl'
if os.path.exists(modelfile):
    os.remove(modelfile)
joblib.dump(model,modelfile)

# Save transformation
minmaxscalarfile = 'minmaxscalar.pckl'
if os.path.exists(minmaxscalarfile):
    os.remove(minmaxscalarfile)
joblib.dump(minmaxscalaer,minmaxscalarfile)
