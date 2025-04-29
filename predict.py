import tensorflow as tf
import keras
import numpy as np

model = keras.models.load_model('vgg16_finetune.keras')
model.summary()

directory_predict = 'archive/tree-bark/train'
file = '/alder/33.jpg'
image_height = 500
image_width = 500
image_size = (image_height, image_width)

img = keras.utils.load_img(directory_predict + file , target_size=(image_height,image_width))
img_array = keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch
predictions = model.predict(img_array)
predicted_classes = np.argmax(predictions, axis=1)
# true_classes = np.argmax(y_test, axis=1)

score = tf.nn.softmax(predictions[0])
print(file)
print("This image most likely belongs to {} with a {:.2f} percent confidence.".format(np.argmax(score), 100 * np.max(score)))



