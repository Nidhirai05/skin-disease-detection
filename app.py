from flask import Flask, request, render_template
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

app = Flask(__name__)
CORS(app)

# Set upload folder
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the trained EfficientNet model
model = tf.keras.models.load_model("SkinDiseaseImageDataset.h5")

# Class index to label mapping
label_map = {
    1: 'Eczema',
    2: 'Warts Molluscum and other Viral Infections',
    3: 'Melanoma',
    4: 'Atopic Dermatitis',
    5: 'Basal Cell Carcinoma (BCC)',
    6: 'Melanocytic Nevi (NV)',
    7: 'Benign Keratosis-like Lesions (BKL)',
    8: 'Psoriasis pictures Lichen Planus and related diseases',
    9: 'Seborrheic Keratoses and other Benign Tumors',
    10: 'Tinea Ringworm Candidiasis and other Fungal Infections'
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file provided", 400

    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    try:
        # Save uploaded image to static folder
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Load and preprocess image
        image = Image.open(filepath).convert('RGB')
        image = image.resize((256, 256))
        img_array = np.array(image)
        img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0] + 1  # adjust to match label_map
        predicted_label = label_map.get(predicted_class, "Unknown")

        confidence = float(np.max(predictions))

        return render_template('result.html',
                               prediction=predicted_label,
                               confidence=f"{confidence:.2f}",
                               image_path=filepath)

    except Exception as e:
        return f"Error occurred: {str(e)}", 500

if __name__ == '__main__':
    app.run(debug=True)
