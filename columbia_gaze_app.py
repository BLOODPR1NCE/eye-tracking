#columbia_gaze_app.py
import cv2
import numpy as np
import tkinter as tk
from tkinter import Canvas
import tensorflow as tf
import time
import sys

class WorkingGazeTracker:
    def __init__(self, model_path='gaze_model_final.keras'):
        print("="*50)
        print("ЗАПУСК ГЛАЗНОГО ТРЕКЕРА")
        print("="*50)
        
        # Загрузка модели
        print("\nЗагрузка модели...")
        try:
            self.model = tf.keras.models.load_model(model_path)
            print("✓ Модель загружена успешно")
        except Exception as e:
            print(f"✗ Ошибка загрузки модели: {e}")
            print("Создание тестовой модели...")
            self.create_dummy_model()
        
        # Загрузка каскадов Хаара
        print("\nЗагрузка детекторов...")
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        
        if self.face_cascade.empty() or self.eye_cascade.empty():
            print("✗ Ошибка: не удалось загрузить каскады!")
            sys.exit(1)
        print("✓ Детекторы загружены")
        
        # Инициализация камеры
        print("\nИнициализация камеры...")
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            print("✗ Ошибка: камера не найдена!")
            sys.exit(1)
        
        # Настройки камеры
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        print("✓ Камера готова")
        
        # Получаем размеры экрана
        self.screen_width = 1920  # Можно изменить под ваш монитор
        self.screen_height = 1080
        
        # Создание окна
        self.root = tk.Tk()
        self.root.title("Eye Gaze Tracker")
        
        # Полноэкранный режим
        self.root.attributes('-fullscreen', True)
        self.root.configure(bg='black')
        
        # Создаем холст
        self.canvas = Canvas(self.root, bg='black', highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Зеленая точка для отслеживания взгляда
        self.dot_size = 25
        self.dot = self.canvas.create_oval(
            0, 0, self.dot_size, self.dot_size,
            fill='#00FF00',
            outline='#00FF00',
            width=3
        )
        
        # Статус
        self.status_text = self.canvas.create_text(
            20, 20,
            text="Инициализация...",
            fill='white',
            font=('Arial', 14, 'bold'),
            anchor='w'
        )
        
        # Флаги
        self.eyes_detected = False
        self.face_detected = False
        
        # Буфер для сглаживания
        self.x_buffer = []
        self.y_buffer = []
        self.buffer_size = 10
        
        # Статистика
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0
        
        # Привязка клавиш
        self.root.bind('<Escape>', self.quit)
        self.root.bind('<c>', self.center_dot)
        self.root.bind('<f>', self.toggle_fullscreen)
        
        # Центрируем точку
        self.center_dot()
        
        # Запуск обновления
        self.update()
    
    def create_dummy_model(self):
        """Создание простой модели для тестирования"""
        print("Создание тестовой модели...")
        left_input = tf.keras.layers.Input(shape=(32, 32, 1))
        right_input = tf.keras.layers.Input(shape=(32, 32, 1))
        
        def process(x):
            x = tf.keras.layers.Flatten()(x)
            x = tf.keras.layers.Dense(16, activation='relu')(x)
            return x
        
        left_features = process(left_input)
        right_features = process(right_input)
        merged = tf.keras.layers.Concatenate()([left_features, right_features])
        output = tf.keras.layers.Dense(2, activation='sigmoid')(merged)
        
        self.model = tf.keras.Model(inputs=[left_input, right_input], outputs=output)
        self.model.compile(optimizer='adam', loss='mse')
        print("✓ Тестовая модель создана")
    
    def detect_eyes(self, frame):
        """Обнаружение лица и глаз с использованием каскадов Хаара"""
        # Конвертируем в grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Улучшаем контраст
        gray = cv2.equalizeHist(gray)
        
        # Находим лица
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5,
            minSize=(150, 150)
        )
        
        if len(faces) == 0:
            self.face_detected = False
            self.eyes_detected = False
            return None, None
        
        # Берем самое большое лицо
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        self.face_detected = True
        
        # Область лица
        face_roi = gray[y:y+h, x:x+w]
        
        # Находим глаза в области лица
        eyes = self.eye_cascade.detectMultiScale(
            face_roi,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(30, 30)
        )
        
        if len(eyes) < 2:
            self.eyes_detected = False
            return None, None
        
        # Сортируем глаза по X координате
        eyes = sorted(eyes, key=lambda e: e[0])
        
        # Берем два глаза (если больше 2, берем первые два)
        if len(eyes) > 2:
            eyes = eyes[:2]
        
        # Извлекаем левый глаз (первый в списке)
        ex1, ey1, ew1, eh1 = eyes[0]
        left_eye = face_roi[ey1:ey1+eh1, ex1:ex1+ew1]
        
        # Извлекаем правый глаз (второй в списке)
        ex2, ey2, ew2, eh2 = eyes[1]
        right_eye = face_roi[ey2:ey2+eh2, ex2:ex2+ew2]
        
        self.eyes_detected = True
        return left_eye, right_eye
    
    def preprocess_eyes(self, left_eye, right_eye):
        """Предобработка глаз для модели"""
        if left_eye is None or right_eye is None:
            return None, None
        
        # Ресайз до 32x32
        left_resized = cv2.resize(left_eye, (32, 32))
        right_resized = cv2.resize(right_eye, (32, 32))
        
        # Улучшение контраста
        left_eq = cv2.equalizeHist(left_resized)
        right_eq = cv2.equalizeHist(right_resized)
        
        # Нормализация
        left_norm = left_eq.astype('float32') / 255.0
        right_norm = right_eq.astype('float32') / 255.0
        
        # Добавляем размерности
        left_processed = np.expand_dims(left_norm, axis=(0, -1))
        right_processed = np.expand_dims(right_norm, axis=(0, -1))
        
        return left_processed, right_processed
    
    def center_dot(self, event=None):
        """Центрирование точки"""
        x = self.screen_width // 2
        y = self.screen_height // 2
        
        # Очищаем буфер
        self.x_buffer = [x] * self.buffer_size
        self.y_buffer = [y] * self.buffer_size
        
        self.move_dot(x, y)
        self.canvas.itemconfig(self.status_text, text="Точка центрирована")
    
    def toggle_fullscreen(self, event=None):
        """Переключение полноэкранного режима"""
        current = self.root.attributes('-fullscreen')
        self.root.attributes('-fullscreen', not current)
    
    def move_dot(self, x, y):
        """Плавное перемещение точки"""
        # Добавляем в буфер
        self.x_buffer.append(x)
        self.y_buffer.append(y)
        
        # Ограничиваем размер буфера
        if len(self.x_buffer) > self.buffer_size:
            self.x_buffer.pop(0)
            self.y_buffer.pop(0)
        
        # Используем медиану для сглаживания
        if len(self.x_buffer) >= 3:
            smooth_x = np.median(self.x_buffer[-3:])
            smooth_y = np.median(self.y_buffer[-3:])
        else:
            smooth_x = np.mean(self.x_buffer) if self.x_buffer else x
            smooth_y = np.mean(self.y_buffer) if self.y_buffer else y
        
        # Плавное движение (экспоненциальное сглаживание)
        current_coords = self.canvas.coords(self.dot)
        if current_coords:
            current_x = (current_coords[0] + current_coords[2]) / 2
            current_y = (current_coords[1] + current_coords[3]) / 2
            
            # Медленное сглаживание для стабильности
            alpha = 0.2
            final_x = current_x * (1 - alpha) + smooth_x * alpha
            final_y = current_y * (1 - alpha) + smooth_y * alpha
        else:
            final_x, final_y = smooth_x, smooth_y
        
        # Ограничиваем границы
        final_x = max(self.dot_size//2, min(self.screen_width - self.dot_size//2, final_x))
        final_y = max(self.dot_size//2, min(self.screen_height - self.dot_size//2, final_y))
        
        # Обновляем позицию
        self.canvas.coords(
            self.dot,
            final_x - self.dot_size//2,
            final_y - self.dot_size//2,
            final_x + self.dot_size//2,
            final_y + self.dot_size//2
        )
        
        return final_x, final_y
    
    def update(self):
        """Основной цикл обновления"""
        # Расчет FPS
        self.frame_count += 1
        elapsed = time.time() - self.start_time
        if elapsed > 1.0:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.start_time = time.time()
        
        # Чтение кадра
        ret, frame = self.cap.read()
        if ret:
            # Обнаружение глаз
            left_eye, right_eye = self.detect_eyes(frame)
            
            # Обновление статуса
            if self.face_detected:
                if self.eyes_detected:
                    status = "✓ Глаза обнаружены"
                    color = "lime"
                    
                    # Предобработка и предсказание
                    left_processed, right_processed = self.preprocess_eyes(left_eye, right_eye)
                    
                    if left_processed is not None:
                        try:
                            prediction = self.model.predict(
                                [left_processed, right_processed], verbose=0
                            )[0]
                            
                            # Преобразуем в координаты экрана
                            x = int(prediction[0] * self.screen_width)
                            y = int(prediction[1] * self.screen_height)
                            
                            # Перемещаем точку
                            final_x, final_y = self.move_dot(x, y)
                            
                            # Обновляем статус
                            status_text = f"FPS: {self.fps:.1f} | Позиция: [{final_x:.0f}, {final_y:.0f}]"
                            
                        except Exception as e:
                            status_text = f"FPS: {self.fps:.1f} | Ошибка предсказания"
                    else:
                        status_text = f"FPS: {self.fps:.1f} | Ошибка обработки"
                else:
                    status = "⚠ Лицо есть, глаза не найдены"
                    color = "yellow"
                    status_text = f"FPS: {self.fps:.1f} | Поиск глаз..."
            else:
                status = "✗ Лицо не обнаружено"
                color = "red"
                status_text = f"FPS: {self.fps:.1f} | Подойдите ближе к камере"
            
            # Обновление интерфейса
            self.canvas.itemconfig(self.status_text, text=status_text)
            
            # Обновление статуса в углу экрана
            self.canvas.create_rectangle(0, 0, 200, 60, fill='black', outline='')
            self.canvas.create_text(100, 30, text=status, fill=color, 
                                   font=('Arial', 12, 'bold'))
        else:
            self.canvas.itemconfig(self.status_text, text="Ошибка камеры")
        
        # Следующее обновление
        self.root.after(33, self.update)  # ~30 FPS
    
    def quit(self, event=None):
        """Выход из приложения"""
        print("\nЗавершение работы...")
        self.cap.release()
        self.root.destroy()
        print("Приложение завершено")
    
    def run(self):
        """Запуск приложения"""
        print("\n" + "="*50)
        print("ИНСТРУКЦИЯ ПО ИСПОЛЬЗОВАНИЮ:")
        print("="*50)
        print("1. Сядьте на расстоянии 50-80 см от камеры")
        print("2. Убедитесь, что ваше лицо хорошо освещено")
        print("3. Расположите лицо в центре кадра")
        print("4. Двигайте ТОЛЬКО ГЛАЗАМИ, не поворачивайте голову")
        print("5. Для калибровки смотрите в центр и нажмите 'C'")
        print("\nУПРАВЛЕНИЕ:")
        print("  ESC - выход")
        print("  C   - центрировать точку")
        print("  F   - переключить полноэкранный режим")
        print("="*50)
        print("\nЗапуск трекера...")
        
        self.root.mainloop()

if __name__ == "__main__":
    try:
        app = WorkingGazeTracker('gaze_model_final.keras')
        app.run()
    except Exception as e:
        print(f"\n✗ Критическая ошибка: {e}")
        print("\nПроверьте следующие моменты:")
        print("1. Установлены все библиотеки (tensorflow, opencv-python)")
        print("2. Камера подключена и работает")
        print("3. Файлы каскадов загружены")
        print("4. Модель обучена и сохранена")
        input("\nНажмите Enter для выхода...")