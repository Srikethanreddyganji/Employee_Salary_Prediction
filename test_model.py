import joblib
import pandas as pd

model = joblib.load("salary_model.pkl")
encoders = joblib.load("encoders.pkl")

test_input = {
    'age': 39,
    'workclass': 'State-gov',
    'fnlwgt': 77516,
    'education': 'Bachelors',
    'educational-num': 13,
    'marital-status': 'Never-married',
    'occupation': 'Adm-clerical',
    'relationship': 'Not-in-family',
    'race': 'White',
    'gender': 'Male',
    'capital-gain': 2174,
    'capital-loss': 0,
    'hours-per-week': 40,
    'native-country': 'United-States'
}

input_df = pd.DataFrame([test_input])

for col in encoders:
    if col != 'target':
        le = encoders[col]
        input_df[col] = le.transform(input_df[col])

prediction = model.predict(input_df)
output_label = encoders['target'].inverse_transform(prediction)

print("Predicted Income Class:", output_label[0])
