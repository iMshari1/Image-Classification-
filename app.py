from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os

app = Flask(__name__)
model = load_model('image_classification_model.h5')


class_names = ['airplane', 'car', 'cat', 'dog', 'fruit', 'person']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file:
        img_path = os.path.join('static', file.filename)
        file.save(img_path)
        
        img = load_img(img_path, target_size=(128, 128))
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = img / 255.0
        
        pred = model.predict(img)
        class_idx = np.argmax(pred)
        class_name = class_names[class_idx]
        
        return render_template('result.html', class_name=class_name, img_path=img_path)

if __name__ == '__main__':
    app.run(debug=True, port=5002)
