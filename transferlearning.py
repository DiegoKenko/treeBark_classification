import keras 
import numpy as np
import matplotlib.pyplot as plt
import os

directory_train = 'archive/tree-bark/train'
image_height = 500
image_width = 500
image_size = (image_height, image_width)
epochs = 5
batch_size = 5
filepath = "model.keras"    
monitoring1 = 'val_accuracy'

def createModel():
    base_model = keras.applications.VGG16(weights='imagenet',input_shape=(image_height, image_width, 3), include_top=False)
    for layer in base_model.layers:
        layer.trainable = False      
        
    inputs = keras.Input(shape=(image_height, image_width, 3))
    x = base_model(inputs, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    outputs = keras.layers.Dense(len(class_names), activation='softmax')(x)
    model = keras.Model(inputs, outputs)    
    model.compile(loss='categorical_crossentropy', optimizer= keras.optimizers.Adam(learning_rate=0.0005), metrics = ['accuracy'])
    model.fit(train_ds, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=val_ds, callbacks=callbacks_list)    
    
    for layer in base_model.layers[-4:]:
        layer.trainable = True
        
    model.compile(optimizer=keras.optimizers.Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_ds, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=val_ds, callbacks=callbacks_list)
    model.save(filepath)
    return model

def continueTraining():
    model = keras.models.load_model(filepath)
    model.compile(loss='categorical_crossentropy', optimizer= keras.optimizers.Adam(learning_rate=0.0005), metrics = ['accuracy'])
    
    # training the model
    history1 = model.fit(train_ds, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=val_ds, callbacks=callbacks_list)
    
    return model


def data_augmentation(x):
    for layer in augmentation_layers:
        x = layer(x)
    return x


(train_ds,val_ds) = keras.preprocessing.image_dataset_from_directory(
    directory_train,
    batch_size=1,
    image_size=image_size,
    subset='both',
    labels='inferred',
    label_mode='categorical',
    validation_split=0.2,
    seed=12)

class_names = train_ds.class_names

augmentation_layers = [
    keras.layers.RandomFlip("horizontal"),
    keras.layers.RandomRotation(0.1),
]

train_ds = train_ds.map(lambda x, y: (data_augmentation(x), y))
# setting model checkpoint, earlystopping, reduce_lr
checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor=monitoring1, verbose=0, save_best_only=True, mode='max')
earlystop = keras.callbacks.EarlyStopping(monitor = monitoring1, min_delta = 0, patience = 4, verbose = 1,restore_best_weights = True)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor=monitoring1, factor=0.2, verbose = 1, patience=2, min_lr=0.000001)
callbacks_list = [checkpoint]
# callbacks_list.append(earlystop)
# callbacks_list.append(reduce_lr)

if os.path.exists(filepath):
    print("Model found, continuing training")
    continueTraining()
else:
    print("Model not found, creating a new one")
    createModel()
    continueTraining()