import tensorflow as tf

# MNIST 데이터셋 로드 및 전처리 수행
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Sequential 모델 및 Dense 레이어 구축
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(28, 28)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# 모델 설정 및 학습 프로세스 실행
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5)

# 테스트 데이터를 통한 최종 성능 평가
model.evaluate(x_test, y_test)