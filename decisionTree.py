import pandas as pd
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

datos = pd.read_csv("train.csv")


# Rellenar los datos nulos
datos["Age"] = datos["Age"].fillna(datos["Age"].mean())
# Quitar los datos que no sean necesarios
datos = datos.drop(["Cabin"], axis=1)
datos = datos.dropna()
datos = datos.drop(["Name", "PassengerId", "Ticket"], axis=1)
# Aplanar los datos en booleanos
dummies_sex = pd.get_dummies(datos["Sex"], dtype=int, drop_first=True)
datos = datos.join(dummies_sex)
datos = datos.drop(["Sex"], axis=1)
# Renombrar columna
datos = datos.rename(columns={'male' : 'sexo'})
dummies_embarked = pd.get_dummies(datos["Embarked"], dtype=int, drop_first=True)
# Aplanar los datos en booleanos
datos = datos.join(dummies_embarked)
datos = datos.drop(["Embarked"], axis=1)

# Variables de prueba
x = datos.drop(["Survived"], axis=1)
y = datos["Survived"]

X_ent, X_pru, Y_ent, Y_pru = train_test_split(x, y, test_size=.2)

# Busqueda de la mejor profundidad para mejor accuracy
better_accuracy = {
    'index':0,
    'accuracy': 0
}

results = []

for i in range(1,15):
    modelo = DecisionTreeClassifier(max_depth=i)
    modelo.fit(X_ent, Y_ent)
    predicciones = modelo.predict(X_pru)
    exactitud = accuracy_score(Y_pru, predicciones)
    if exactitud > better_accuracy['accuracy']:
        better_accuracy['accuracy'] = exactitud
        better_accuracy["index"] = i
    results.append(exactitud)
    print(f"La exactitud del modelo con depth {i} es: {exactitud}")

# Grafica de las distintas accuracy
sb.lineplot(data=results)
plt.show()

# Modelo con mejor accuracy
modelo = DecisionTreeClassifier(max_depth=better_accuracy["index"])
modelo.fit(X_ent, Y_ent)
predicciones = modelo.predict(X_pru)
exactitud = accuracy_score(Y_pru, predicciones)
print(exactitud)