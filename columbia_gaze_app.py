# gaze_tracker_simple.py
import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
import tensorflow as tf

class SimpleGazeTracker:
    def __init__(self):
        self.model = tf.keras.models.load_model('gaze_model_simple.keras')
        self.cap = cv2.VideoCapture(0)
        
        # Создание GUI
        self.root = tk.Tk()
        self.root.title("Gaze Tracker")
        
        self.canvas = tk.Canvas(self.root, width=1920, height=1080)
        self.canvas.pack()
        
        # Точка взгляда
        self.dot = self.canvas.create_oval(0, 0, 20, 20, fill='red')
        
        # Запуск
        self.update()
        self.root.mainloop()
    
    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        left, right = self.detect_eyes(gray)
        
        if left is not None and right is not None:
            # Предсказание
            left_input = self.prepare_eye(left)[np.newaxis, ..., np.newaxis]
            right_input = self.prepare_eye(right)[np.newaxis, ..., np.newaxis]
            
            pred = self.model.predict([left_input, right_input], verbose=0)[0]
            
            # Обновление точки
            x, y = int(pred[0] * 1920), int(pred[1] * 1080)
            self.canvas.coords(self.dot, x-10, y-10, x+10, y+10)
        
        # Показ видео
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        img_tk = ImageTk.PhotoImage(image=img)
        
        if hasattr(self, 'video_label'):
            self.video_label.config(image=img_tk)
            self.video_label.image = img_tk
        else:
            self.video_label = tk.Label(self.root, image=img_tk)
            self.video_label.image = img_tk
            self.video_label.pack()
        
        return frame
    
    def detect_eyes(self, img):
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        faces = face_cascade.detectMultiScale(img, 1.1, 5)
        
        if len(faces) == 0:
            return None, None
            
        x, y, w, h = faces[0]
        face_roi = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(face_roi, 1.1, 3)
        
        if len(eyes) < 2:
            return None, None
            
        eyes = sorted(eyes, key=lambda e: e[0])
        ex1, ey1, ew1, eh1 = eyes[0]
        ex2, ey2, ew2, eh2 = eyes[1]
        
        left = face_roi[ey1:ey1+eh1, ex1:ex1+ew1]
        right = face_roi[ey2:ey2+eh2, ex2:ex2+ew2]
        
        return left, right
    
    def prepare_eye(self, eye):
        eye = cv2.resize(eye, (36, 36))
        eye = cv2.equalizeHist(eye)
        return eye.astype('float32') / 255.0
    
    def update(self):
        self.process_frame()
        self.root.after(30, self.update)
    
    def __del__(self):
        self.cap.release()

if __name__ == "__main__":
    app = SimpleGazeTracker()