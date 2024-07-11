import os
from flask import Flask, request, render_template, redirect, send_from_directory
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing.image import img_to_array, load_img # type: ignore
import numpy as np
import tensorflow as tf

# Menyembunyikan pesan log TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

tf.config.set_soft_device_placement(True)

# Memuat model TFLite
interpreter = tf.lite.Interpreter(model_path='ai_model/model3.tflite')
interpreter.allocate_tensors()

# Mendapatkan detail tensor input dan output
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_and_augment_image(image_path, target_size=(224, 224)):
    image = load_img(image_path, target_size=target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    
    return image


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        img_array = preprocess_and_augment_image(file_path)
        
        input_data = np.array(img_array, dtype=np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        
        interpreter.invoke()
        
        output_data = interpreter.get_tensor(output_details[0]['index'])
        predicted_class = np.argmax(output_data, axis=1)[0]
        class_labels = {0: 'Bad', 1: 'Good', 2: 'Mixed'}
        
        return render_template('predict.html', context={
            'result': class_labels[predicted_class],
            'img': file_path,
        })

    return 'File not allowed'

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.run(debug=False, host='0.0.0.0', port=8000)
