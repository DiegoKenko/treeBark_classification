import numpy as np
import tensorflow as tf
import keras
from keras import layers, models, losses
import matplotlib.pyplot as plt

directory_train = 'archive/tree-bark/train'
directory_test = 'archive/tree-bark/test'
directory_predict = 'archive/tree-bark/validate'
image_height = 500
image_width = 500
image_size = (image_height, image_width)
epochs = 5

ds = keras.preprocessing.image_dataset_from_directory(
    directory_train,
    labels='inferred',
    batch_size=32,
    image_size=image_size,
    shuffle=True,
    seed=1234,
    interpolation='bilinear',
    follow_links=False
)
ds_teste = keras.preprocessing.image_dataset_from_directory(
    directory_train,
    labels='inferred',
    batch_size=32,
    image_size=image_size,
    shuffle=True,
    seed=1234,
    interpolation='bilinear',
    follow_links=False
)

class_names = ds.class_names

AUTOTUNE = tf.data.AUTOTUNE

ds = ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
ds_teste = ds_teste.cache().prefetch(buffer_size=AUTOTUNE)

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
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2,seed=4),
    layers.RandomZoom(0.1,seed=4),
  ]
)

model = models.Sequential([
    data_augmentation,
    layers.Rescaling(1./255),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(class_names), name="outputs")
])

    
model.compile(optimizer='adam',loss=losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])
history = model.fit(ds, epochs=epochs, validation_data=ds_teste)    

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

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
test_loss, test_acc = model.evaluate(ds_teste, verbose=2)

model.summary()
model.save('model.keras')

img = keras.utils.load_img(directory_predict + '/birch/1.jpg', target_size=(image_height,image_width))
img_array = keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch
predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])
print("This image most likely belongs to {} with a {:.2f} percent confidence.".format(class_names[np.argmax(score)], 100 * np.max(score)))