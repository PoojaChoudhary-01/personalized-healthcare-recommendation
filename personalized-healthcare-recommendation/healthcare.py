#laod dataset

import pandas as pd
df = pd.read_csv('C:/Users/POOJA/Downloads/gitdemo/personalized-healthcare-recommendation/Healthcare dataset.csv')

print(df.head())
print(df.info())
print(df.describe())

#data cleaning and preprocessing

from sklearn.preprocessing import LabelEncoder
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
X = df.iloc[:, :-1]         # Features
y = df.iloc[:, -1]          # Target

#splitting

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#model evaluation 
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=20, max_depth=10, random_state=42)
model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report 
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

#plotting
import matplotlib.pyplot as plt

features = X.columns
scores = model.feature_importances_

plt.bar(features, scores)
plt.xticks(rotation=45)
plt.title("Top Features")
plt.xlabel("Features")
plt.ylabel("Score")
plt.show()

