import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

print("--- Step 1: Loading Raw Data ---")
# Load the simulated faulty dataset from non_real_data folder
df = pd.read_csv('non_real_data/raw_smartwatch_data.csv')
print(f"Original dataset shape (with duplicates & NaNs): {df.shape}")

print("\n--- Step 2: Handling Duplicate Entries ---")
# Remove identical rows caused by duplicate injection
df_cleaned = df.drop_duplicates()
print(f"Dataset shape after removing duplicates: {df_cleaned.shape}")

print("\n--- Step 3: Preprocessing Categorical Data for ML ---")
# Convert 'Motion_Status' (text) into numerical features using One-Hot Encoding
# Machine Learning models require numerical inputs
df_encoded = pd.get_dummies(df_cleaned, columns=['Motion_Status'], drop_first=False)

# Identify feature columns (X) and target column (y)
# Features: Accelerometer data, steps, and the encoded motion statuses
feature_cols = [col for col in df_encoded.columns if col not in ['Timestamp', 'Heart_Rate_BPM']]

print("\n--- Step 4: Splitting Data into Train and Predict Sets ---")
# Split rows into two: where Heart_Rate is known (Train) and where it is missing (Predict)
train_data = df_encoded[df_encoded['Heart_Rate_BPM'].notna()]
predict_data = df_encoded[df_encoded['Heart_Rate_BPM'].isna()]

X = train_data[feature_cols]
y = train_data['Heart_Rate_BPM']

# Split the known data temporarily to evaluate the model performance
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n--- Step 5: Training the Machine Learning Model ---")
# Using Random Forest Regressor to predict the continuous Heart Rate values
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Validate the model
y_pred = model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
r2 = r2_score(y_val, y_pred)
print(f"Model Evaluation -> Validation RMSE: {rmse:.2f} BPM")
print(f"Model Evaluation -> R² Score (Accuracy): {r2:.4f}")

print("\n--- Step 6: Predicting and Filling Missing (NaN) Heart Rates ---")
# Retrain on the entire known dataset before final prediction
model.fit(X, y)

# Predict the missing values
X_missing = predict_data[feature_cols]
predicted_heart_rates = model.predict(X_missing)

# Insert the predicted values back into the missing spots of our cleaned DataFrame
df_cleaned.loc[df_cleaned['Heart_Rate_BPM'].isna(), 'Heart_Rate_BPM'] = predicted_heart_rates

print("All missing Heart Rate values successfully predicted and imputed!")
print(f"Final dataset shape (Cleaned & Completed): {df_cleaned.shape}")

# Save the fully processed and clean data
df_cleaned.to_csv('ingestion/cleaned_smartwatch_data.csv', index=False)
print("Processed data saved to 'ingestion/cleaned_smartwatch_data.csv'")