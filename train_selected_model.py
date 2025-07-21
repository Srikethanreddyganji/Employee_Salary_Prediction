import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import joblib

df = pd.read_csv("adult 3.csv")

selected_features = ["education", "occupation", "gender", "hours-per-week"]
target = "income"

df = df[selected_features + [target]].dropna()

label_encoders = {}
for col in ["education", "occupation", "gender"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

target_encoder = LabelEncoder()
df[target] = target_encoder.fit_transform(df[target])

X = df[selected_features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = GradientBoostingClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

joblib.dump(model, "salary_model.pkl")
joblib.dump(label_encoders, "encoders.pkl")
joblib.dump(target_encoder, "target_encoder.pkl")

print("Model and encoders saved successfully")
