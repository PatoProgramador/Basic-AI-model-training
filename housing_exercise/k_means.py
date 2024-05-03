import pandas as pd
import seaborn as sb
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

datos = pd.read_csv("housing.csv")

X = datos.loc[:,["latitude", "longitude", "median_income"]]

model = KMeans(n_clusters=6)
predict = model.fit_predict(X)

predict.shape

X["segmento_economico"] = predict

# sb.scatterplot(x="longitude", y="latitude", data=X, hue="segmento_economico", palette="coolwarm", size="segmento_economico", sizes=(20,50))
# sb.countplot(x="segmento_economico", data=X)
print(X.groupby(["segmento_economico"])["median_income"].mean())