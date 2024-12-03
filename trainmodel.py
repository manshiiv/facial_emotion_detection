from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
import os
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

TRAIN_DIR = 'images/train'
TEST_DIR = 'images/test'

def createdataframe(dir):
    image_paths = []
    labels = []
    for label in os.listdir(dir):
        label_path = os.path.join(dir, label)
        if os.path.isdir(label_path):
            for imagename in os.listdir(label_path):
                image_paths.append(os.path.join(label_path, imagename))
                labels.append(label)
            print(label, "completed")
    return image_paths, labels

# Create the dataframe
train_image_paths, train_labels = createdataframe(TRAIN_DIR)
test_image_paths, test_labels = createdataframe(TEST_DIR)

train = pd.DataFrame({'image': train_image_paths, 'label': train_labels})
test = pd.DataFrame({'image': test_image_paths, 'label': test_labels})

# Encode the labels
label_encoder = LabelEncoder()
train['label'] = label_encoder.fit_transform(train['label'])
test['label'] = label_encoder.transform(test['label'])

# One-hot encode the labels
y_train = to_categorical(train['label'], num_classes=7)
y_test = to_categorical(test['label'], num_classes=7)

def extract_features(image_paths):
    features = []
    for image in tqdm(image_paths):
        img = load_img(image, color_mode='grayscale')
        img = np.array(img)
        features.append(img)
    return np.array(features)

# Extract features
train_features = extract_features(train['image'])
test_features = extract_features(test['image'])

x_train = train_features
x_test = test_features

model = Sequential()
# convolutional layers
model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(48,48,1)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(128, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(256, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(512, kernel_size=(1,1), activation='relu'))
model.add(Flatten())
# fully connected layers
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
# output layer
model.add(Dense(7, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Fit the model
model.fit(x=x_train, y=y_train, batch_size=128, epochs=10, validation_data=(x_test, y_test))

model.save('model.h5')
