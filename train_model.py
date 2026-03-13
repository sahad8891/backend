import pandas as pd
import joblib
from model import TWSM

data = pd.read_csv("../dataset/autism_screening.csv")

X = data.drop("Class", axis=1).values
y = data["Class"].values

model = TWSM()
model.fit(X,y)

joblib.dump(model,"models/twsm_model.pkl")

print("Model trained successfully")
