import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

df = pd.read_csv(r"C:\Users\muham\Downloads\archive (12)\Diabetes Health Indicators.csv")


df = df[~df["DIABETE4"].isin([7, 9])]

df["Diabetes_binary"] = df["DIABETE4"].apply(lambda x: 1 if x == 1 else 0)


y = df["Diabetes_binary"]
X = df.drop(["DIABETE4", "Diabetes_binary"], axis=1)

X = X.fillna(X.mean())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


model = LogisticRegression(
    solver='liblinear',
    C=1.0,
    max_iter=1000,
    random_state=42
)


model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

plt.figure()
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

labels = [["TN", "FP"],
          ["FN", "TP"]]

for i in range(2):
    for j in range(2):
        plt.text(j, i,
                 f"{labels[i][j]}\n{cm[i][j]}",
                 ha="center",
                 va="center")

plt.xticks([0,1], ["No Diabetes", "Diabetes"])
plt.yticks([0,1], ["No Diabetes", "Diabetes"])

plt.show()
