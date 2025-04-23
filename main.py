import joblib
import json
import pandas as pd

# Load model and scaler
model = joblib.load("best_rf_model.pkl")
scaler = joblib.load("scaler.pkl")

# Load the model's expected feature columns
with open("model_columns.json", "r") as f:
    model_columns = json.load(f)

# Sample raw input (should include all raw fields before preprocessing)
sample_raw = {
    'Age': 23,
    'Gender': 'Other',
    'City': 'Urban',
    'Profession': 'Student',
    'Degree': 'PhD',
    'Academic Pressure': 5,
    'Work Pressure': 2,
    'CGPA': 9.4,
    'Study Satisfaction': 3,
    'Job Satisfaction': 4,
    'Sleep Duration': 1,
    'Work/Study Hours': 9,
    'Have you ever had suicidal thoughts ?': 'No',
    'Family History of Mental Illness': 'No',
    'Financial Stress': 'Moderate'
}

# Total Pressure
sample_raw['Total Pressure'] = sample_raw['Academic Pressure'] + sample_raw['Work Pressure']

# Convert to DataFrame
sample_df = pd.DataFrame([sample_raw])

# Apply one-hot encoding as in training
categorical_features = ['Gender', 'City', 'Profession', 'Degree',
                        'Have you ever had suicidal thoughts ?', 'Family History of Mental Illness', 'Financial Stress']
sample_df = pd.get_dummies(sample_df, columns=categorical_features, drop_first=True)

# Scale numeric features
numeric_features = ['Age', 'Academic Pressure', 'Work Pressure', 'CGPA', 'Study Satisfaction', 'Job Satisfaction',
                    'Sleep Duration', 'Work/Study Hours', 'Total Pressure']
sample_df[numeric_features] = scaler.transform(sample_df[numeric_features])

# Add missing columns with 0s all at once
missing_cols = [col for col in model_columns if col not in sample_df.columns]
missing_df = pd.DataFrame(0, index=sample_df.index, columns=missing_cols)

# Combine existing and missing columns
sample_df = pd.concat([sample_df, missing_df], axis=1)

# Reorder columns to match training data
sample_df = sample_df[model_columns]

# Predict
prediction = model.predict(sample_df)
print("Predicted class:", prediction[0])
