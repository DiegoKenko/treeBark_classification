import numpy as np
import tensorflow as tf
import keras
from keras import layers, models, losses
import matplotlib.pyplot as plt

filepath = 'model.keras'
directory_train = 'archive/tree-bark/train'
image_height = 800
image_width = 800
image_size = (image_height, image_width)
epochs = 50
batch_size = 2

(train_ds,val_ds) = keras.preprocessing.image_dataset_from_directory(
    directory_train,
    batch_size=1,
    image_size=image_size,
    subset='both',
    validation_split=0.2,
    seed=12,
)


class_names = train_ds.class_names

# plt.figure(figsize=(10, 10))
# for images, labels in ds.take(1):
#   for i in range(9):
#     ax = plt.subplot(3, 3, i + 1)
#     plt.imshow(images[i].numpy().astype("uint8"))
#     plt.title(class_names[labels[i]])
#     plt.axis("off")
# plt.show()

data_augmentation = keras.Sequential(
  [
    layers.RandomFlip("horizontal", input_shape=(image_height, image_width, 3)),
    layers.RandomRotation(0.2),
    layers.RandomZoom(height_factor=(0.1,0.3),width_factor=(0.1,0.3)),
  ]
)

model = keras.models.Sequential()
model.add(data_augmentation)
model.add(layers.Conv2D(64, (3,3), strides=(1,1), padding='same', activation = 'relu', input_shape=(image_height,image_width,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Dropout(0.10))
model.add(layers.Conv2D(32, (3,3), strides=(1,1), padding='same', activation = 'relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Dropout(0.1))
model.add(layers.Flatten())
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(len(class_names), activation='softmax'))
model.compile(optimizer='adam',loss=losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])

checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]


new_model = keras.models.load_model(filepath)
np.testing.assert_allclose(model.predict(train_ds),new_model.predict(train_ds))
new_model.summary()

checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
new_model.fit(train_ds, epochs=5, batch_size=batch_size, callbacks=callbacks_list)
acc = new_model.history['accuracy']
val_acc = new_model.history['val_accuracy']

loss = new_model.history['loss']
val_loss = new_model.history['val_loss']

epochs_range = range(epochs)

# plt.figure(figsize=(8, 8))
# plt.subplot(1, 2, 1)
# plt.plot(epochs_range, acc, label='Training Accuracy')
# plt.plot(epochs_range, val_acc, label='Validation Accuracy')
# plt.legend(loc='lower right')
# plt.title('Training and Validation Accuracy')

# plt.subplot(1, 2, 2)
# plt.plot(epochs_range, loss, label='Training Loss')
# plt.plot(epochs_range, val_loss, label='Validation Loss')
# plt.legend(loc='upper right')
# plt.title('Training and Validation Loss')
# plt.show()
test_loss, test_acc = model.evaluate(train_ds, verbose=2)
