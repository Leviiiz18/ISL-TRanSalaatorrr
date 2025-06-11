import numpy as np
import tensorflow as tf
from keras.models import load_model
from tensorflow import keras
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras import layers, models

# --- Settings ---
train_dir = 'Indian'  # Your training data folder path
img_height, img_width = 64, 64
batch_size = 32
num_classes = 35
temperature = 5.0  # temperature for softening probabilities

# Load your trained teacher model
teacher_model = load_model('teacher_model.h5')

# Prepare data generator (same preprocessing as training)
datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False  # keep order for matching soft targets with data
)

# Function to apply temperature scaling to probabilities
def apply_temperature_scaling(probs, temperature):
    logits = np.log(np.clip(probs, 1e-20, 1.0))  # Avoid log(0)
    scaled_logits = logits / temperature
    exp_logits = np.exp(scaled_logits)
    softened_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    return softened_probs

# Collect soft targets and true labels
soft_targets_list = []
labels_list = []

num_batches = len(train_generator)

for i in range(num_batches):
    X_batch, y_batch = train_generator[i]
    probs = teacher_model.predict(X_batch, verbose=0)  # Suppress progress bar
    soft_targets = apply_temperature_scaling(probs, temperature)

    print(f"\nBatch {i+1}/{num_batches} - Softmax Output (before temperature scaling):\n", probs)
    print(f"Batch {i+1}/{num_batches} - Soft Targets (after temperature scaling):\n", soft_targets)

    soft_targets_list.append(soft_targets)
    labels_list.append(y_batch)

# Combine all batches into arrays
all_soft_targets = np.vstack(soft_targets_list)
all_labels = np.vstack(labels_list)

print("\nFinal Soft Targets (first 5 samples):")
print(all_soft_targets[:5])

print("\nFinal True Labels (first 5 samples):")
print(all_labels[:5])

np.save('soft_targets.npy', all_soft_targets)
np.save('true_labels.npy', all_labels)