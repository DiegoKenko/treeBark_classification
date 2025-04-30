import tensorflow as tf
import keras
import numpy as np

model = keras.models.load_model('model.keras')
model.summary()
directory_predict = 'archive/tree-bark/train'
file = '/Cedrus/IMG_6162.jpg'
image_height = 500
image_width = 500
image_size = (image_height, image_width)

img = keras.utils.load_img(directory_predict + file , target_size=(image_height,image_width))
img_array = keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, axis=0) # Create a batch
img_array = keras.applications.vgg19.preprocess_input(img_array)
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions,  axis=1)
# true_classes = np.argmax(y_test, axis=1)

score = tf.nn.softmax(predictions[0])
print(file)
print(predicted_class[0])
print(100 * np.max(score), '% confidence that this is a', predicted_class[0])



