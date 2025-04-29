import keras 
import numpy as np
import matplotlib.pyplot as plt

directory_train = 'archive/tree-bark/train'
image_height = 500
image_width = 500
image_size = (image_height, image_width)
epochs = 5
batch_size = 32

(train_ds,val_ds) = keras.preprocessing.image_dataset_from_directory(
    directory_train,
    batch_size=1,
    image_size=image_size,
    subset='both',
    label_mode='categorical',
    validation_split=0.2,
    seed=12,
)
class_names = train_ds.class_names

base_model = keras.applications.ResNet50V2(
    include_top=False,
    weights='imagenet',
    input_shape=(image_height, image_width, 3),
    classes=len(class_names),
)

model = keras.Sequential()
model.add(base_model) 
#Adding the Dense layers along with activation and batch normalization
model.add(keras.layers.Dense(1000, activation='relu'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dropout(rate=0.5, seed=42))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(500, activation='relu'))
model.add(keras.layers.Dropout(rate=0.5, seed=42))
model.add(keras.layers.Dense(len(class_names), activation='softmax'))

model.compile(
    loss='categorical_crossentropy', 
    optimizer= keras.optimizers.Adam(learning_rate=0.0005), 
    metrics = ['accuracy']
)

monitoring1 = 'val_accuracy'
filepath="newModel.keras"

# setting model checkpoint, earlystopping, reduce_lr
checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor=monitoring1, verbose=0, save_best_only=True, mode='max')
earlystop = keras.callbacks.EarlyStopping(monitor = monitoring1, min_delta = 0, patience = 4, verbose = 1,restore_best_weights = True)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor=monitoring1, factor=0.2, verbose = 1, patience=2, min_lr=0.000001)
callbacks_list = [checkpoint, earlystop, reduce_lr]

# Setting meximum epochs
epochs = 200

# training the model
history1 = model.fit(train_ds, batch_size=256, epochs=epochs, verbose=1, validation_data=val_ds,shuffle=True, callbacks=callbacks_list)

plt.figure(0)
plt.plot(history1.history['accuracy'], label='training accuracy')
plt.plot(history1.history['val_accuracy'], label='val accuracy')
plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()

plt.figure(1)
plt.plot(history1.history['loss'], label='training loss')
plt.plot(history1.history['val_loss'], label='val loss')
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()