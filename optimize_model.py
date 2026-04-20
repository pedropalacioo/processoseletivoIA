import tensorflow as tf
import os

def main():
    model = tf.keras.models.load_model('model.h5')

    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()

    # Salvando em Disco
    with open('model.tflite', 'wb') as f:
        f.write(tflite_model)

    print("Modelo otimizado salvo como 'model.tflite'")

    h5_size = os.path.getsize('model.h5') / (1024 * 1024) # MB
    tflite_size = os.path.getsize('model.tflite') / (1024 * 1024) # MB

    print(f"Modelo original (H5): {h5_size:.2f} MB")
    print(f"Modelo otimizado (TFLite): {tflite_size:.2f} MB")
    print(f"Redução: {((h5_size - tflite_size) / h5_size * 100):.1f}%")

if __name__ == "__main__":
    main()

