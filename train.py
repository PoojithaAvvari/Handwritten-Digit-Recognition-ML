import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, datasets, models

# ================================
# 1. Load & Prepare Dataset
# ================================
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

# Reshape for CNN input (28x28x1)
train_images = train_images.reshape((60000, 28, 28, 1)).astype("float32") / 255.0
test_images = test_images.reshape((10000, 28, 28, 1)).astype("float32") / 255.0

print("TRAIN IMAGES:", train_images.shape)
print("TEST IMAGES:", test_images.shape)

# ================================
# 2. Define Model
# ================================
def create_model():
    model = keras.Sequential([
        layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')  # <-- Softmax for multi-class classification
    ])

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy']
    )
    return model

# ================================
# 3. Train or Load Model
# ================================
MODEL_PATH = "trained_model.h5"

if os.path.exists(MODEL_PATH):
    print(f"Loading saved model from {MODEL_PATH}...")
    model = models.load_model(MODEL_PATH)
else:
    print("No saved model found. Training a new one...")
    model = create_model()
    history = model.fit(train_images, train_labels, epochs=10, batch_size=32)
    
    # Plot training metrics
    acc = history.history['accuracy']
    loss = history.history['loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(8, 6))
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.legend(loc='lower right')
    plt.title('Training Accuracy & Loss')
    plt.show()

    # Save model
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

# ================================
# 4. Single Image Prediction
# ================================
image = train_images[1].reshape(1, 28, 28, 1)
model_pred = np.argmax(model.predict(image), axis=-1)
plt.imshow(image.reshape(28, 28), cmap='gray')
plt.title(f"Predicted: {model_pred[0]}")
plt.show()
print(f"Prediction of model: {model_pred[0]}")

# ================================
# 5. Multiple Image Prediction
# ================================
images = test_images[1:5]
print("Test images array shape:", images.shape)

plt.figure(figsize=(8, 4))
for i, test_image in enumerate(images, start=1):
    prediction = np.argmax(model.predict(test_image.reshape(1, 28, 28, 1)), axis=-1)
    plt.subplot(1, 4, i)
    plt.axis('off')
    plt.title(f"Pred: {prediction[0]}")
    plt.imshow(test_image.reshape(28, 28), cmap='gray')
plt.show()
