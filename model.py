import pandas as pd
import joblib
from sklearn.naive_bayes import GaussianNB

df = pd.read_csv("kyphosis.csv")

X = df[["Age", "Number", "Start"]]
y = df["Kyphosis"]

kyp = GaussianNB() 
kyp.fit(X, y)


joblib.dump(kyp, "kyp.pkl")