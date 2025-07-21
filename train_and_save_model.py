import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv("adult 3.csv")

selected_features = ['age', 'workclass', 'fnlwgt', 'education', 'educational-num', 
                     'marital-status', 'occupation', 'relationship', 'race', 
                     'gender', 'capital-gain', 'capital-loss', 
                     'hours-per-week', 'native-country']
target_column = 'income'

X = df[selected_features]
y = df[target_column]

encoders = {}

for col in X.select_dtypes(include='object').columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    encoders[col] = le

le_target = LabelEncoder()
y = le_target.fit_transform(y)
encoders['target'] = le_target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = GradientBoostingClassifier()
model.fit(X_train, y_train)

joblib.dump(model, "salary_model.pkl")
joblib.dump(encoders, "encoders.pkl")
print("Model and encoders saved successfully")
