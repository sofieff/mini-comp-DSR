import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()


import pandas as pd
import numpy as np

# select features we want
features = [
    "reanalysis_specific_humidity_g_per_kg",
    "reanalysis_dew_point_temp_k",
    "station_avg_temp_c",
    "station_min_temp_c",
]

X = sj_train_features[features][:idx_sj]
# X = X_sj_train_features[:][:idx_sj]
y = sj_train_features['total_cases'][:idx_sj]

# X[features] = scale.fit_transform(X[features].values)

# Add a constant column to our model so we can have a Y-intercept
X = sm.add_constant(X)

print (X)

est = sm.OLS(y, X).fit()

print(est.summary())



# Predicting on the test set
X = sj_train_features[features][idx_sj+1:]
# X = X_sj_train_features[:][idx_sj+1:]

# X = sj_test_features[features]
# X[features] = scale.fit_transform(X[features].values)

X = sm.add_constant(X)

print(X)
predicted = est.predict(X)
print(predicted)