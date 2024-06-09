from PIL import Image

import tensorflow as tf
import numpy as np
from matplotlib import image
from matplotlib import pyplot as plt
from tensorflow import keras
#tf.config.set_visible_devices([],'GPU')


def generate_img(image):
    img_size = 120
    generator = tf.keras.models.load_model('fina_dolor_model_125.h5',compile=False)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    generator.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy')

    # Load the grayscale test image
    rgb_img = image
    original_size = rgb_img.size
    rgb_image = rgb_img.resize((img_size, img_size))
    rgb_img_array = (np.asarray(rgb_image)) / 255
    gray_image = rgb_image.convert('L')
    gray_img_array = (np.asarray(gray_image).reshape((1, img_size, img_size, 1))) / 255
    
    # Generate colorized image
    colorized_image = generator.predict(gray_img_array)
    colorized_img = Image.fromarray((colorized_image[0] * 255).astype('uint8'))
    resized_colorized_img = colorized_img.resize((120,120))
    
        
    return resized_colorized_img

