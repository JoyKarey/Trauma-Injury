import tensorflow as tf
from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
from io import BytesIO
import configparser

app = Flask(__name__)

# Load the Keras model from a .keras directory
model = tf.keras.models.load_model('best_model.h5')

# Define a data preprocessing function for images
def preprocess_image(image):
    # Load the image using Pillow (PIL)
    img = Image.open(image)

    # Resize the image to match your model's input size
    target_size = (224, 224)  # Adjust to match your model's expected input size
    img = img.resize(target_size)

    # Convert the image to a NumPy array
    img_array = np.array(img)

    # Normalize the image data (if required)
    img_array = img_array / 255.0  # Example normalization for values in the [0, 255] range

    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'data' not in request.files:
            return jsonify({'error': 'No image file provided'})

        image = request.files['data']
        if image.filename == '':
            return jsonify({'error': 'No selected file'})

        input_data = preprocess_image(image)
        predictions = model.predict(np.array([input_data]))

        return jsonify({'predictions': predictions.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run()
