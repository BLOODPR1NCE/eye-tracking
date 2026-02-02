# gaze_tracker_app.py
import cv2
import numpy as np
import tkinter as tk
from tkinter import Canvas, Label, Frame
import tensorflow as tf
import time
import sys
import os
from PIL import Image, ImageTk
import threading

class AdvancedGazeTracker:
    def __init__(self, model_path='models/gaze_model_final.keras'):
        print("="*60)
        print("ЗАПУСК ПРОДВИНУТОГО ГЛАЗНОГО ТРЕКЕРА")
        print("="*60)
        
        self.model = None #инициализация
        self.cap = None
        self.running = True
        self.face_detected = False
        self.eyes_detected = False
        self.current_prediction = (0.5, 0.5)
        
        self.screen_width = 1920 #экран
        self.screen_height = 1080
        self.dot_size = 30
        self.smoothing_factor = 0.3
        
        self.x_buffer = []
        self.y_buffer = []
        self.buffer_size = 15
        
        self.frame_count = 0 #статистика
        self.start_time = time.time()
        self.fps = 0
        self.detection_history = []
        
        print("\nЗагрузка модели...")
        if not self.load_model(model_path):
            print("Не удалось загрузить модель. Используется демо-режим.")
        
        print("Загрузка детекторов...")
        self.load_detectors()
        
        print("Инициализация камеры...")
        self.init_camera()
        
        print("Создание интерфейса...")
        self.create_gui()
        
        self.start_threads()
    
    def load_model(self, model_path):
        """Загружает модель"""
        try:
            if os.path.exists(model_path):
                self.model = tf.keras.models.load_model(model_path)
                print(f"✓ Модель загружена: {model_path}")
                return True
            else:
                possible_paths = [
                    'gaze_model_final.keras',
                    'gaze_model_final.h5',
                    'models/gaze_model_final.h5'
                ]
                
                for path in possible_paths:
                    if os.path.exists(path):
                        self.model = tf.keras.models.load_model(path)
                        print(f"✓ Модель загружена: {path}")
                        return True
                
                print("✗ Модель не найдена")
                return False
                
        except Exception as e:
            print(f"✗ Ошибка загрузки модели: {e}")
            return False
    
    def load_detectors(self):
        """Загружает детекторы лиц и глаз"""
        try:
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            self.eye_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_eye.xml'
            )
            
            if self.face_cascade.empty() or self.eye_cascade.empty():
                print("✗ Не удалось загрузить детекторы")
                return False
            
            print("✓ Детекторы загружены")
            return True
            
        except Exception as e:
            print(f"✗ Ошибка загрузки детекторов: {e}")
            return False
    
    def init_camera(self):
        """Инициализирует камеру"""
        try:
            for i in range(3):
                self.cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
                if self.cap.isOpened():
                    print(f"✓ Камера найдена (индекс {i})")
                    
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640) #настройка камеры
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    self.cap.set(cv2.CAP_PROP_FPS, 30)
                    
                    self.camera_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) #размеры камеры
                    self.camera_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    
                    print(f"  Размер: {self.camera_width}x{self.camera_height}")
                    return True
            
            print("✗ Камера не найдена")
            return False
            
        except Exception as e:
            print(f"✗ Ошибка инициализации камеры: {e}")
            return False
    
    def create_gui(self):
        self.root = tk.Tk() #основное окно
        self.root.title("Advanced Gaze Tracker")
        self.root.configure(bg='#2c3e50')
        self.root.attributes('-fullscreen', True)
        
        self.canvas = Canvas(self.root, bg='#34495e', highlightthickness=0) #холст для взгляда
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Зеленая точка для отслеживания
        self.dot = self.canvas.create_oval(
            0, 0, self.dot_size, self.dot_size,
            fill='#2ecc71',
            outline='#27ae60',
            width=3
        )
        
        self.info_frame = Frame(self.root, bg='#2c3e50') #инфопанель
        self.info_frame.place(x=20, y=20)
        
        # Элементы информации
        self.status_label = Label(self.info_frame, 
                                 text="Статус: Инициализация...",
                                 font=('Arial', 12, 'bold'),
                                 fg='white',
                                 bg='#2c3e50')
        self.status_label.pack(anchor='w')
        
        self.fps_label = Label(self.info_frame,
                              text="FPS: 0.0",
                              font=('Arial', 10),
                              fg='#ecf0f1',
                              bg='#2c3e50')
        self.fps_label.pack(anchor='w')
        
        self.position_label = Label(self.info_frame,
                                   text="Позиция: [0, 0]",
                                   font=('Arial', 10),
                                   fg='#ecf0f1',
                                   bg='#2c3e50')
        self.position_label.pack(anchor='w')
        
        self.mode_label = Label(self.info_frame,
                               text="Режим: Загрузка...",
                               font=('Arial', 10),
                               fg='#ecf0f1',
                               bg='#2c3e50')
        self.mode_label.pack(anchor='w')
        
        # Видео панель
        self.video_label = Label(self.root, bg='#2c3e50')
        self.video_label.place(x=20, y=150)
        
        # Привязка клавиш
        self.root.bind('<Escape>', self.quit)
        self.root.bind('<c>', self.calibrate)
        self.root.bind('<f>', self.toggle_fullscreen)
        self.root.bind('<r>', self.reset_tracking)
        self.root.bind('<d>', self.toggle_debug)
        
        # Центрируем точку
        self.center_dot()
        
        # Флаг отладки
        self.debug_mode = False
    
    def center_dot(self, event=None):
        x = self.screen_width // 2
        y = self.screen_height // 2
        
        self.x_buffer = [x] * self.buffer_size
        self.y_buffer = [y] * self.buffer_size
        
        self.move_dot(x, y)
        self.status_label.config(text="Статус: Точка центрирована", fg='#2ecc71')
    
    def calibrate(self, event=None):
        self.status_label.config(text="Статус: Калибровка...", fg='#f39c12')
        self.center_dot()
        self.status_label.config(text="Статус: Калибровка завершена", fg='#2ecc71')
    
    def toggle_fullscreen(self, event=None):
        current = self.root.attributes('-fullscreen')
        self.root.attributes('-fullscreen', not current)
    
    def reset_tracking(self, event=None):
        self.x_buffer = []
        self.y_buffer = []
        self.center_dot()
        self.status_label.config(text="Статус: Трекинг сброшен", fg='#e74c3c')
    
    def toggle_debug(self, event=None):
        self.debug_mode = not self.debug_mode
        mode = "ВКЛ" if self.debug_mode else "ВЫКЛ"
        self.status_label.config(text=f"Статус: Режим отладки {mode}", fg='#9b59b6')
    
    def process_frame(self, frame):
        self.frame_count += 1 #фпс
        elapsed = time.time() - self.start_time
        if elapsed > 1.0:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.start_time = time.time()
            self.fps_label.config(text=f"FPS: {self.fps:.1f}")
        
        left_eye, right_eye, processed_frame = self.detect_eyes(frame) #обнаружение глаз
        
        if left_eye is not None and right_eye is not None:
            self.eyes_detected = True
            
            if self.model is not None:
                prediction = self.predict_gaze(left_eye, right_eye)
                self.current_prediction = prediction
                
                x = int(prediction[0] * self.screen_width)
                y = int(prediction[1] * self.screen_height)
                
                x, y = self.apply_smoothing(x, y)
                
                self.move_dot(x, y)
                
                self.position_label.config(text=f"Позиция: [{x}, {y}]")
                self.status_label.config(text="Статус: Отслеживание активно", fg='#2ecc71')
        else:
            self.eyes_detected = False
            self.status_label.config(text="Статус: Глаза не обнаружены", fg='#e74c3c')
        
        # Обновляем режим
        mode_text = "Режим: Нейросеть" if not self.demo_mode else "Режим: Демо"
        if self.debug_mode:
            mode_text += " (отладка)"
        self.mode_label.config(text=mode_text)
        
        return processed_frame
    
    def detect_eyes(self, frame):
        """Обнаруживает глаза на кадре"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        
        display_frame = frame.copy()
        
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(100, 100)
        )
        
        if len(faces) == 0:
            self.face_detected = False
            return None, None, display_frame
        
        self.face_detected = True
        
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        face_roi = gray[y:y+h, x:x+w]
        
        if self.debug_mode:
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        eyes = self.eye_cascade.detectMultiScale(
            face_roi,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(30, 30)
        )
        
        if len(eyes) < 2:
            return None, None, display_frame
        
        eyes = sorted(eyes, key=lambda e: e[0])

        ex1, ey1, ew1, eh1 = eyes[0]
        ex2, ey2, ew2, eh2 = eyes[1]
        
        left_eye = face_roi[ey1:ey1+eh1, ex1:ex1+ew1]
        right_eye = face_roi[ey2:ey2+eh2, ex2:ex2+ew2]
        
        left_eye = self.preprocess_eye(left_eye)
        right_eye = self.preprocess_eye(right_eye)
        
        if self.debug_mode:
            cv2.rectangle(display_frame, 
                         (x + ex1, y + ey1), 
                         (x + ex1 + ew1, y + ey1 + eh1), 
                         (255, 0, 0), 2)
            cv2.rectangle(display_frame, 
                         (x + ex2, y + ey2), 
                         (x + ex2 + ew2, y + ey2 + eh2), 
                         (255, 0, 0), 2)
        
        return left_eye, right_eye, display_frame
    
    def preprocess_eye(self, eye_img):
        if eye_img is None or eye_img.size == 0:
            return None

        eye_resized = cv2.resize(eye_img, (36, 36))
        eye_eq = cv2.equalizeHist(eye_resized)
        eye_norm = eye_eq.astype('float32') / 255.0
        
        return eye_norm
    
    def predict_gaze(self, left_eye, right_eye):
        if left_eye is None or right_eye is None:
            return (0.5, 0.5)
        
        try:
            left_input = np.expand_dims(left_eye, axis=(0, -1))
            right_input = np.expand_dims(right_eye, axis=(0, -1))
            prediction = self.model.predict([left_input, right_input], verbose=0)[0]
            
            return tuple(prediction)
            
        except Exception as e:
            print(f"Ошибка предсказания: {e}")
            return (0.5, 0.5)
    
    def apply_smoothing(self, x, y):
        """Применяет сглаживание к координатам"""
        self.x_buffer.append(x)
        self.y_buffer.append(y)
        
        if len(self.x_buffer) > self.buffer_size:
            self.x_buffer.pop(0)
            self.y_buffer.pop(0)
        
        smooth_x = 0
        smooth_y = 0
        
        for i in range(len(self.x_buffer)):
            weight = self.smoothing_factor * (1 - self.smoothing_factor) ** i
            smooth_x += self.x_buffer[-(i+1)] * weight
            smooth_y += self.y_buffer[-(i+1)] * weight
        
        total_weight = sum(self.smoothing_factor * (1 - self.smoothing_factor) ** i #номарлизация
                          for i in range(len(self.x_buffer)))
        
        if total_weight > 0:
            smooth_x /= total_weight
            smooth_y /= total_weight
        
        return int(smooth_x), int(smooth_y)
    
    def move_dot(self, x, y):
        x = max(self.dot_size//2, min(self.screen_width - self.dot_size//2, x))
        y = max(self.dot_size//2, min(self.screen_height - self.dot_size//2, y))
        
        self.canvas.coords(
            self.dot,
            x - self.dot_size//2,
            y - self.dot_size//2,
            x + self.dot_size//2,
            y + self.dot_size//2
        )
    
    def update_video(self):
        while self.running:
            if self.cap is not None and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    processed_frame = self.process_frame(frame)
                    
                    rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(rgb_frame)
                    img_tk = ImageTk.PhotoImage(image=img)
                    
                    self.video_label.config(image=img_tk)
                    self.video_label.image = img_tk
                else:
                    print("Ошибка чтения кадра")
            
            time.sleep(0.03)
    
    def start_threads(self):
        self.video_thread = threading.Thread(target=self.update_video, daemon=True)
        self.video_thread.start()
    
    def quit(self, event=None):
        print("\nЗавершение работы...")
        self.running = False
        
        if self.cap is not None:
            self.cap.release()
        
        self.root.quit()
        self.root.destroy()
        print("Приложение завершено")
    
    def run(self):
        """Запускает приложение"""
        print("\n" + "="*60)
        print("ИНСТРУКЦИЯ ПО ИСПОЛЬЗОВАНИЮ:")
        print("="*60)
        print("1. Сядьте на расстоянии 50-80 см от камеры")
        print("2. Убедитесь в хорошем освещении")
        print("3. Расположите лицо в центре кадра")
        print("4. Для калибровки смотрите в центр и нажмите 'C'")
        print("\nУПРАВЛЕНИЕ:")
        print("  ESC - выход")
        print("  C   - калибровка")
        print("  F   - полноэкранный режим")
        print("  R   - сброс трекинга")
        print("  D   - режим отладки")
        print("="*60)
        print("\nЗапуск трекера...")
        
        try:
            self.root.mainloop()
        except Exception as e:
            print(f"\nОшибка в приложении: {e}")
        finally:
            self.quit()

if __name__ == "__main__":
    try:
        model_path = 'models/gaze_model_final.keras'
        if not os.path.exists(model_path):
            print(f"Модель не найдена: {model_path}")
            print("Завершение работы...")
        else:
            app = AdvancedGazeTracker(model_path)
            app.run()
            
    except Exception as e:
        print(f"\nКритическая ошибка: {e}")
        import traceback
        traceback.print_exc()
        input("\nНажмите Enter для выхода...")