import pandas as pd
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier

datos = pd.read_csv("train.csv")

datos["Age"] = datos["Age"].fillna(datos["Age"].mean())
datos = datos.drop(["Cabin"], axis=1)
datos = datos.dropna()
datos = datos.drop(["Name", "PassengerId", "Ticket"], axis=1)
dummies_sex = pd.get_dummies(datos["Sex"], dtype=int, drop_first=True)
datos = datos.join(dummies_sex)
datos = datos.drop(["Sex"], axis=1)
datos = datos.rename(columns={'male' : 'sexo'})
dummies_embarked = pd.get_dummies(datos["Embarked"], dtype=int, drop_first=True)
datos = datos.join(dummies_embarked)
datos = datos.drop(["Embarked"], axis=1)

x = datos.drop(["Survived"], axis=1)
y = datos["Survived"]


X_ent, X_pru, Y_ent, Y_pru = train_test_split(x, y, test_size=.2)

modelo = DecisionTreeClassifier(max_depth=10)
modelo.fit(X_ent, Y_ent)
predicciones = modelo.predict(X_pru)

print(accuracy_score(Y_pru, predicciones))