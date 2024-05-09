import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import math

datos, metadatos = tfds.load("mnist", as_supervised=True, with_info=True)

datos_train = datos["train"]
datos_test = datos["test"]

class_names = metadatos.features["label"].names

# Normalizacion
# 0-255 - Pixel
# Transformar 0-255 a 0-1
def normalizer(images, tags):
    images = tf.cast(images, tf.float32)
    images = images/255
    return images, tags

datos_train = datos_train.map(normalizer)
datos_test = datos_test.map(normalizer)

# Agregar cache
datos_train = datos_train.cache()
datos_test = datos_test.cache()

plt.figure(figsize=(10,10))
for i, (images, tags) in enumerate(datos_train.take(25)):
    # Presentar
    plt.subplot(5, 5, i+1)
    plt.imshow(images, cmap=plt.cm.binary)
    
plt.show()

model = tf.keras.Sequential([
    # 28x28 Pixeles = 784 datos de entrada
    tf.keras.layers.Flatten(input_shape=(28,28,1)), # 1 = Blanco y negro
    # Capa Oculta
    tf.keras.layers.Dense(50, activation=tf.nn.relu),
    tf.keras.layers.Dense(50, activation=tf.nn.relu),
    # Capa de salida
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"]
)

size_slot = 32

datos_train = datos_train.repeat().shuffle(60000).batch(size_slot)
datos_test = datos_test.batch(size_slot)

# Entrenar modelo
train = model.fit(
    datos_train, epochs=10,
    steps_per_epoch=math.ceil(60000/size_slot)
)

