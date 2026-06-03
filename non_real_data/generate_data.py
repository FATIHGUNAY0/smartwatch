import pandas as pd
import numpy as np

# 1. Random Seed and Time Series Configuration (24 hours of data, every 10 seconds)
np.random.seed(42)
timestamps = pd.date_range(start="2026-06-03 00:00:00", periods=8640, freq="10s")
n_samples = len(timestamps)

motion_status = []
accelerometer_g = []
step_count_delta = []
heart_rate = []

# 2. Simulating Sensor Data Based on 4 Different Motion Statuses
for i in range(n_samples):
    # Creating a daily life routine scenario:
    
    if i < 2160:  # From 00:00 to 06:00: Sleeping
        status = "Sleeping"
        acc = np.random.normal(1.0, 0.01)    # Static, almost no movement
        steps = 0                             # No steps during sleep
        hr = np.random.normal(58, 3)         # Low resting heart rate
        
    elif i < 4320:  # From 06:00 to 12:00: Normal Activity
        status = "Normal"
        acc = np.random.normal(1.1, 0.08)    # Minor movements (sitting, standing)
        steps = np.random.choice([0, 1, 2, 3], p=[0.7, 0.15, 0.1, 0.05]) # Occasional small steps
        hr = np.random.normal(72, 5)         # Average resting heart rate
        
    elif i < 6480:  # From 12:00 to 18:00: Walking
        status = "Walking"
        acc = np.random.normal(1.4, 0.15)   # Clear rhythmic movement patterns
        steps = np.random.randint(6, 13)     # Steady walking steps
        hr = np.random.normal(98, 8)         # Active heart rate
        
    else:  # From 18:00 to 24:00: Workout
        status = "Workout"
        acc = np.random.normal(2.6, 0.5)     # Intense and rapid movements
        steps = np.random.randint(16, 26)    # Running or high-intensity exercise steps
        hr = np.random.normal(148, 12)       # High workout heart rate
        
    motion_status.append(status)
    accelerometer_g.append(acc)
    step_count_delta.append(steps)
    heart_rate.append(hr)

# 3. Converting Simulated Data into a Pandas DataFrame
df = pd.DataFrame({
    'Timestamp': timestamps,
    'Motion_Status': motion_status,
    'Accelerometer_G': accelerometer_g,
    'Step_Delta': step_count_delta,
    'Heart_Rate_BPM': heart_rate
})

# 4. Project Requirement: Injecting Missing Values (NaN) and Duplicates
# Randomly drop 12% of the heart rate values (To be predicted later using ML)
nan_indices = np.random.choice(df.index, size=int(n_samples * 0.12), replace=False)
df.loc[nan_indices, 'Heart_Rate_BPM'] = np.nan

# Explicitly inject 20 duplicate rows into the dataset
duplicates = df.sample(20)
df = pd.concat([df, duplicates], ignore_index=True).sort_values('Timestamp')

# 5. Exporting the Raw Data to CSV
df.to_csv('non_real_data/raw_smartwatch_data.csv', index=False)
print("Raw data successfully generated and saved to 'non_real_data/raw_smartwatch_data.csv'!")
print("\nFirst 10 rows of the generated dataset:")
print(df.head(10))