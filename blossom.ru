import tensorflow as tf
from tensorflow.keras import layers, models

# Загрузка данных: 60 000 картинок цифр 28x28 для обучения
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Модель: свёрточная сеть — стандарт для распознавания картинок
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')  # 10 классов: цифры 0–9
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5, batch_size=64)

# Проверяем точность на тестовых данных
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Точность на тесте: {test_acc:.4f}")

# Сохраняем обученную модель
model.save('digit_model.keras')
print("Модель сохранена в digit_model.keras")
