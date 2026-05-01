"""
Retrain model with current sklearn version and save with joblib
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor
import joblib

# Read data
data = pd.read_csv('../data/renfe_trains.csv')
df = pd.DataFrame(data)

# Clean data
df = df[df['price'] != 'price']
df['price'] = df['price'].astype(np.float32)
df.dropna(subset=['vehicle_class', 'price', 'fare'], how='all', inplace=True)
df.fillna({'price': df['price'].mean()}, inplace=True)
df.drop_duplicates(inplace=True)

# Identify numeric, categorical, and datetime columns
y = df[['price']].astype(float)
X = df.drop(['price'], axis=1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define encoding columns
nominal_cols = ['destination', 'fare']
ordinal_cols = ['vehicle_class']
drop_cols = ['company', 'origin', 'departure', 'arrival']  # Drop non-numeric columns that aren't being encoded

# Drop columns that aren't useful for prediction
X_train = X_train.drop(drop_cols, axis=1)
X_test = X_test.drop(drop_cols, axis=1)

# Encode categorical features
nominal_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

# Fit and transform training data
X_train_nominal = nominal_encoder.fit_transform(X_train[nominal_cols])
X_train_encoded_cols = nominal_encoder.get_feature_names_out(nominal_cols)
X_train = X_train.drop(nominal_cols, axis=1).reset_index(drop=True)
X_train = pd.concat([X_train, pd.DataFrame(X_train_nominal, columns=X_train_encoded_cols)], axis=1)

X_train_ordinal = ordinal_encoder.fit_transform(X_train[ordinal_cols])
X_train[ordinal_cols] = X_train_ordinal

# Transform test data
X_test_nominal = nominal_encoder.transform(X_test[nominal_cols])
X_test = X_test.drop(nominal_cols, axis=1).reset_index(drop=True)
X_test = pd.concat([X_test, pd.DataFrame(X_test_nominal, columns=X_train_encoded_cols)], axis=1)

X_test_ordinal = ordinal_encoder.transform(X_test[ordinal_cols])
X_test[ordinal_cols] = X_test_ordinal

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train.values.ravel())

# Evaluate
from sklearn.metrics import mean_squared_error, r2_score
y_pred = model.predict(X_test)
print(f"R² Score: {r2_score(y_test, y_pred)}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred))}")

# Save model and encoders with joblib (better compatibility)
joblib.dump(model, 'model.joblib')
joblib.dump(nominal_encoder, 'nominal.joblib')
joblib.dump(ordinal_encoder, 'ordinal.joblib')
joblib.dump(X_train.columns.tolist(), 'feature_names.joblib')

print("Model retrained and saved successfully!")
