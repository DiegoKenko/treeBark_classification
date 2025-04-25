import keras 
import numpy as np

directory_train = 'archive/tree-bark/train'
directory_test = 'archive/tree-bark/test'
directory_validate = 'archive/tree-bark/validate'
image_height = 500
image_width = 500
image_size = (image_height, image_width)
epochs = 5
batch_size = 32

ds = keras.preprocessing.image_dataset_from_directory(
    directory_train,
    label_mode='categorical',
    labels='inferred',
    batch_size=1,
    color_mode='rgb',
    image_size=image_size,
    shuffle=True,
    seed=1234,
    subset='training',
    validation_split=0.2,
)

ds_val = keras.preprocessing.image_dataset_from_directory(
    directory_validate,
    labels='inferred',
    label_mode='categorical',
    batch_size=1,
    color_mode='rgb',
    image_size=image_size,
    shuffle=True,
    subset='validation',
    validation_split=0.2,
    seed=1234,
)

vgg_model = keras.applications.VGG16(
    include_top=False,
    weights='imagenet',
    input_shape=(image_height, image_width, 3),
    pooling='avg'
)

# Freeze four convolution blocks
for layer in vgg_model.layers[:15]:
    layer.trainable = False
    
# Make sure you have frozen the correct layers
for i, layer in enumerate(vgg_model.layers):
    print(i, layer.name, layer.trainable)

vgg_model.summary()

x = vgg_model.output
x = keras.layers.Flatten()(x) # Flatten dimensions to for use in FC layers
x = keras.layers.Dense(512, activation='relu')(x)
x = keras.layers.Dropout(0.5)(x) # Dropout layer to reduce overfitting
x = keras.layers.Dense(256, activation='relu')(x)
x = keras.layers.Dense(12, activation='softmax')(x) # Softmax for multiclass

transfer_model = keras.Model(inputs=vgg_model.input, outputs=x)

transfer_model.summary()

lr_reduce = keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.6, patience=8, verbose=1, mode='max', min_lr=5e-5)
checkpoint = keras.callbacks.ModelCheckpoint('vgg16_finetune.keras', monitor= 'val_accuracy', mode= 'max', save_best_only = True, verbose= 1)
transfer_model.compile(loss="categorical_crossentropy", optimizer=keras.optimizers.Adam(learning_rate=5e-5), metrics=["accuracy"])
history = transfer_model.fit(ds, batch_size = batch_size, epochs=epochs, validation_data=ds_val, callbacks=[lr_reduce,checkpoint])

for layer in vgg_model.layers[:15]:
    layer.trainable = False
    