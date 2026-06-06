import importlib
import tensorflow as tf

try:
    tfds = importlib.import_module("tensorflow_datasets")
except ImportError as e:
    raise ImportError(
        "tensorflow_datasets is required to load the EMNIST dataset. "
        "Install it with `pip install tensorflow-datasets`."
    ) from e

layers = tf.keras.layers
models = tf.keras.models

# Load EMNIST Letters via tensorflow_datasets
# as_supervised=True returns (image, label). batch_size=-1 loads entire split as a single tensor.
ds_train, ds_test = tfds.load('emnist/letters', split=['train', 'test'], as_supervised=True, batch_size=-1)

# ds_train and ds_test are tuples of (images, labels)
x_train, y_train = tfds.as_numpy(ds_train)
x_test, y_test = tfds.as_numpy(ds_test)

# Normalize and ensure channel dimension
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

if x_train.ndim == 3:
    x_train = x_train.reshape((-1, 28, 28, 1))
if x_test.ndim == 3:
    x_test = x_test.reshape((-1, 28, 28, 1))

# CNN Model
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(26, activation='softmax')
])

# Compile
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train
model.fit(x_train, y_train, epochs=5)

# Save model
model.save("model.h5")

# Evaluate
test_loss, test_acc = model.evaluate(x_test, y_test)

print("Accuracy:", test_acc)