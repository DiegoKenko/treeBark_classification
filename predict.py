import tensorflow as tf
import keras
import numpy as np

model = keras.models.load_model('model.keras')
model.summary()

directory_predict = 'archive/tree-bark/validate'
image_height = 256
image_width = 256
image_size = (image_height, image_width)

img = keras.utils.load_img(directory_predict + '/birch/1.jpg', target_size=(image_height,image_width))
img_array = keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch
predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])
print("This image most likely belongs to {} with a {:.2f} percent confidence.".format(predictions[0], 100 * np.max(score)))