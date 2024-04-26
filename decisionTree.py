import pandas as pd

datos = pd.read_csv("train.csv")

print(datos.head())

print(datos.describe())