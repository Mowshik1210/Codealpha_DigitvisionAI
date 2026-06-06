import tensorflow as tf

datasets = tf.keras.datasets
layers = tf.keras.layers
models = tf.keras.models

# 1. Load dataset
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

# 2. Normalize Pixel Channels
x_train = x_train / 255.0
x_test = x_test / 255.0

# 3. Reshape for Channel Vectors
x_train = x_train.reshape((-1, 28, 28, 1))
x_test = x_test.reshape((-1, 28, 28, 1))

# 4. Deepened CNN Model Architecture (Upgraded for 99%+ Peak Performance)
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2)),
    
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2)),
    
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.BatchNormalization(),
    layers.Flatten(),
    
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),  # Reduces training noise overfitting
    layers.Dense(10, activation='softmax')
])

# 5. Compile with Adaptive Learning Optimizer
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 6. Train for more epochs to unlock accuracy gains
print("Starting high-accuracy deep learning training routine...")
model.fit(x_train, y_train, epochs=8, batch_size=64, validation_split=0.1)

# 7. Save model
model.save("model.h5")

# 8. Evaluate Final System Weights
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"\n🏆 Final Validation Accuracy Achieved: {test_acc * 100:.2f}%")