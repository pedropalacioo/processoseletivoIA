import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Download e carregamento do MNIST
(x_train, y_train),(x_test, y_test) = keras.datasets.mnist.load_data()

# Normalização
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Adição de canal
x_train = x_train[..., None]
x_test = x_test[..., None]

# Construção da CNN
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),  # 26×26 para 13×13
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),  # 11×11 para 5×5
    layers.Conv2D(128, (3, 3), activation="relu"),  # 3×3
    layers.Flatten(),  # 1152 valores
    layers.Dense(128, activation="relu"),
    layers.Dense(10, activation="softmax")  # Classificação
])

# Compilação
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Treinamento
print("Iniciando treinamento...")
model.fit(
    x_train, y_train,
    epochs=5,
    batch_size=128,
    validation_data=(x_test, y_test)
)

# Acurácia
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"\n Acurácia: {test_accuracy * 100:.2f}%")

# Salvamento
model.save("model.h5")
print("Modelo salvo em model.h5")