# gaze_api.py
from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import cv2
import base64
import os

app = Flask(__name__)

print("Загрузка модели...")
model_path = 'models/gaze_model_final.keras'
if os.path.exists(model_path):
    model = tf.keras.models.load_model(model_path)
    print("Модель загружена")
else:
    model = None
    print("Модель не найдена!")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model_loaded': model is not None})

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    file = request.files['image']
    img_bytes = file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return jsonify({'error': 'Invalid image'}), 400
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    
    if len(faces) == 0:
        return jsonify({'error': 'No face detected'}), 400
    
    x, y, w, h = faces[0]
    face_roi = gray[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(face_roi, 1.1, 3)
    
    if len(eyes) < 2:
        return jsonify({'error': 'Eyes not detected'}), 400
    
    eyes = sorted(eyes, key=lambda e: e[0])
    ex1, ey1, ew1, eh1 = eyes[0]
    ex2, ey2, ew2, eh2 = eyes[1]
    
    left = face_roi[ey1:ey1+eh1, ex1:ex1+ew1]
    right = face_roi[ey2:ey2+eh2, ex2:ex2+ew2]
    
    if left.size == 0 or right.size == 0:
        return jsonify({'error': 'Eyes empty'}), 400
    
    try:
        left = cv2.resize(left, (36, 36))
        right = cv2.resize(right, (36, 36))
        left = cv2.equalizeHist(left).astype('float32') / 255.0
        right = cv2.equalizeHist(right).astype('float32') / 255.0
    except:
        return jsonify({'error': 'Processing error'}), 400
    
    left_input = left.reshape(1, 36, 36, 1)
    right_input = right.reshape(1, 36, 36, 1)
    
    pred = model.predict([left_input, right_input], verbose=0)[0]
    
    return jsonify({
        'gaze_point': {
            'x': float(pred[0]),
            'y': float(pred[1])
        }
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)