import numpy as np
from keras import layers, models, optimizers, losses
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split


# --- Settings ---
data_dir = 'Indian'  # Same directory used to generate teacher outputs
img_height, img_width = 64, 64
batch_size = 32
num_classes = 35

# --- Load soft targets and true labels ---
soft_targets = np.load('soft_targets.npy')
true_labels = np.load('true_labels.npy')

# --- Load corresponding input images again using ImageDataGenerator ---
datagen = ImageDataGenerator(rescale=1./255)

image_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False  # To match saved soft targets
)

# Extract all images from generator into a numpy array
X_data = []
for i in range(len(image_generator)):
    X_batch, _ = image_generator[i]
    X_data.append(X_batch)

X_data = np.vstack(X_data)  # shape: (num_samples, 64, 64, 3)

# --- Split into training and validation sets ---
X_train, X_val, y_soft_train, y_soft_val = train_test_split(
    X_data, soft_targets, test_size=0.2, random_state=42
)

# --- Define the Student Model ---
def build_student_model(input_shape=(64, 64, 3), num_classes=35):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

student_model = build_student_model()

# --- Compile using KL Divergence for soft targets ---
student_model.compile(
    optimizer=optimizers.Adam(),
    loss=losses.KLDivergence(),  # Knowledge distillation loss
    metrics=['accuracy']
)

# --- Train the Student Model ---
student_model.fit(
    X_train, y_soft_train,
    validation_data=(X_val, y_soft_val),
    epochs=10,
    batch_size=32
)

# --- Save Student Model ---
student_model.save("student_model1.h5")
