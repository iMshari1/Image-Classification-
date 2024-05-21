import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from collections import Counter


data_dir = '/Users/mshari/Documents/ImageClassificationApp/archive/natural_images'


class_names = ['airplane', 'car', 'cat', 'dog', 'fruit', 'person']

images = []
labels = []

for class_name in class_names:
    class_dir = os.path.join(data_dir, class_name)
    for img_name in os.listdir(class_dir):
        img_path = os.path.join(class_dir, img_name)
        img = load_img(img_path, target_size=(128, 128))
        img = img_to_array(img)
        images.append(img)
        labels.append(class_names.index(class_name))

images = np.array(images, dtype="float32") / 255.0
labels = np.array(labels)


def balance_data(images, labels):
    class_counts = Counter(labels)
    min_count = min(class_counts.values())

    balanced_images = []
    balanced_labels = []

    for class_name in class_names:
        class_idx = class_names.index(class_name)
        class_images = images[labels == class_idx]
        class_labels = labels[labels == class_idx]

        if len(class_images) > min_count:
            class_images = class_images[:min_count]
            class_labels = class_labels[:min_count]

        balanced_images.append(class_images)
        balanced_labels.append(class_labels)

    balanced_images = np.vstack(balanced_images)
    balanced_labels = np.hstack(balanced_labels)

    return balanced_images, balanced_labels

balanced_images, balanced_labels = balance_data(images, labels)


X_train, X_temp, y_train, y_temp = train_test_split(balanced_images, balanced_labels, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)


y_train = to_categorical(y_train, num_classes=len(class_names))
y_val = to_categorical(y_val, num_classes=len(class_names))
y_test = to_categorical(y_test, num_classes=len(class_names))


model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(len(class_names), activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

datagen.fit(X_train)


history = model.fit(datagen.flow(X_train, y_train, batch_size=32),
                    epochs=15, validation_data=(X_val, y_val))

test_loss, test_acc = model.evaluate(X_test, y_test)


print(f"Test Accuracy: {test_acc * 100:.2f}%")


y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

cm = confusion_matrix(y_true_classes, y_pred_classes)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

model.save('image_classification_model.h5')