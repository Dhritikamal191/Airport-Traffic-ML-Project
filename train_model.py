# ===============================
# ?? IMPORT LIBRARIES
# ===============================
import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.metrics import mean_squared_error, r2_score

from xgboost import XGBRegressor

# ===============================
# ?? LOAD DATA
# ===============================
df = pd.read_csv("airport_traffic_2025.csv")
df.dropna(inplace=True)

df['FLT_DATE'] = pd.to_datetime(df['FLT_DATE'])

# ===============================
# ?? FEATURE ENGINEERING
# ===============================
df['YEAR'] = df['FLT_DATE'].dt.year
df['MONTH'] = df['FLT_DATE'].dt.month
df['DAY'] = df['FLT_DATE'].dt.day
df['WEEKDAY'] = df['FLT_DATE'].dt.weekday
df['IS_WEEKEND'] = (df['WEEKDAY'] >= 5).astype(int)

df['DEP_ARR_RATIO'] = df['FLT_DEP_1'] / (df['FLT_ARR_1'] + 1)
df['IFR_RATIO'] = df['FLT_TOT_IFR_2'] / (df['FLT_TOT_1'] + 1)

# ===============================
# ?? TARGET & FEATURES
# ===============================
y = df['FLT_TOT_1']

X = df[[
    'YEAR', 'MONTH', 'DAY', 'WEEKDAY', 'IS_WEEKEND',
    'APT_ICAO', 'STATE_NAME',
    'DEP_ARR_RATIO', 'IFR_RATIO'
]]

# ===============================
# ?? SPLIT
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================
# ?? PREPROCESSOR
# ===============================
num_features = [
    'YEAR', 'MONTH', 'DAY', 'WEEKDAY',
    'IS_WEEKEND', 'DEP_ARR_RATIO', 'IFR_RATIO'
]

cat_features = ['APT_ICAO', 'STATE_NAME']

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_features),
    ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), cat_features)
])

# ===============================
# ?? MODEL PIPELINE
# ===============================
model = Pipeline([
    ('preprocessor', preprocessor),
    ('model', XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    ))
])

# ===============================
# ??? TRAIN
# ===============================
model.fit(X_train, y_train)

# ===============================
# ?? EVALUATE
# ===============================
y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("RMSE:", rmse)
print("R2:", r2)

# ===============================
# ?? SAVE SINGLE PKL
# ===============================
with open("xgb_airport_pipeline.pkl", "wb") as f:
    pickle.dump(model, f)

print("? Saved: xgb_airport_pipeline.pkl")