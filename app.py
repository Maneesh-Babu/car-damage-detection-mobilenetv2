from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
model = load_model('model.h5')

CLASS_NAMES = ['Damaged', 'Whole']

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    pred = model.predict(img_array)
    return CLASS_NAMES[np.argmax(pred)]

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        file = request.files['file']
        filepath = os.path.join('static', file.filename)
        file.save(filepath)
        prediction = predict_image(filepath)
        return render_template('index.html', prediction=prediction, image_path=filepath)
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
