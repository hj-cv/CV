import tensorflow as tf
import numpy as np
from PIL import Image
import os

# CIFAR-10 데이터셋 로드 및 전처리
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# CNN 기반 분류 모델 구축
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(32, 32, 3)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# 모델 설정 및 학습 수행
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10)

# 모델 성능 평가 및 외부 이미지 예측
model.evaluate(x_test, y_test)
base_path = os.path.dirname(os.path.abspath(__file__))
img_path = os.path.join(base_path, '..', 'dog.jpg')
img = Image.open(img_path).resize((32, 32))
img_array = np.array(img) / 255.0
prediction = model.predict(np.expand_dims(img_array, axis=0))
print(np.argmax(prediction))