import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

datos = pd.read_csv('celsius.csv')

X = datos["celsius"].values
y = datos["fahrenheit"].values

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1,)), # La entrada, los datos celsius: 1 solo dato
    tf.keras.layers.Dense(units=1) # La salida, es un solo dato, los grados fahrenheit
])

model.compile(
    optimizer= tf.keras.optimizers.Adam(0.1),
    loss="mean_squared_error"
)

training = model.fit(X, y, epochs=1000)

plt.xlabel("# Epoca")
plt.ylabel("Magnitud de perdida")
plt.plot(training.history["loss"])

# prediccion
c = 18
predict = model.predict([[c]])
print(f"Prediccion: {c} son {predict[0]} fahrenheit")

print(model.layers[0].get_weights())