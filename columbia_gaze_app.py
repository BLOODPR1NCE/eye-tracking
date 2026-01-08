# columbia_gaze_app.py
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
from PIL import Image, ImageTk
import numpy as np
import torch
from torchvision import transforms
import os

# Импортируем нашу модель
from columbia_model_training import EfficientGazeNet

class ColumbiaGazeApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Columbia Gaze Tracking System")
        self.root.geometry("1200x700")
        
        # Загрузка модели
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        
        # Инициализация переменных
        self.camera_active = False
        self.cap = None
        self.current_image = None
        
        # Создание интерфейса
        self.setup_ui()
        
    # В методе load_model замените загрузку модели:

    def load_model(self, model_path='columbia_gaze_model.pt'):
        """Загрузка обученной модели"""
        model = EfficientGazeNet()
        try:
            if os.path.exists(model_path):
                # Способ 1: Загружаем только веса с weights_only=True
                model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
                model.to(self.device)
                model.eval()
                print("Columbia модель успешно загружена (только веса)")
            
                # Альтернативно, если нужна обратная совместимость:
                # checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
                # model.load_state_dict(checkpoint['model_state_dict'])
            else:
                print("Файл модели не найден. Используется случайная инициализация.")
        except Exception as e:
            print(f"Ошибка при загрузке модели: {e}")
            print("Попытка загрузки с weights_only=False...")
            try:
                # Попробуем загрузить старым способом
                if os.path.exists(model_path):
                    checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
                    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['model_state_dict'])
                    else:
                        model.load_state_dict(checkpoint)
                    model.to(self.device)
                    model.eval()
                    print("Columbia модель успешно загружена (старый формат)")
            except Exception as e2:
                print(f"Ошибка при загрузке модели старым способом: {e2}")
                print("Используется случайно инициализированная модель")
    
        return model
    
    def setup_ui(self):
        """Настройка пользовательского интерфейса"""
        # Создание панели управления
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        # Заголовок
        title_label = ttk.Label(
            control_frame,
            text="Columbia Gaze Tracking System",
            font=('Helvetica', 16, 'bold')
        )
        title_label.grid(row=0, column=0, columnspan=4, pady=(0, 20))
        
        # Кнопки выбора режима
        ttk.Button(
            control_frame,
            text="📷 Загрузить фото",
            command=self.load_image_mode,
            width=20
        ).grid(row=1, column=0, padx=5, pady=5)
        
        ttk.Button(
            control_frame,
            text="🎥 Запустить камеру",
            command=self.camera_mode,
            width=20
        ).grid(row=1, column=1, padx=5, pady=5)
        
        ttk.Button(
            control_frame,
            text="⏹️ Остановить камеру",
            command=self.stop_camera,
            width=20
        ).grid(row=1, column=2, padx=5, pady=5)
        
        ttk.Button(
            control_frame,
            text="ℹ️ Информация",
            command=self.show_info,
            width=20
        ).grid(row=1, column=3, padx=5, pady=5)
        
        # Область отображения
        self.display_frame = ttk.Frame(self.root, padding="10")
        self.display_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Изображение с камеры/фото
        self.image_label = ttk.Label(self.display_frame)
        self.image_label.grid(row=0, column=0, padx=10, pady=10)
        
        # Область визуализации взгляда
        self.gaze_canvas = tk.Canvas(
            self.display_frame,
            width=800,
            height=600,
            bg='white'
        )
        self.gaze_canvas.grid(row=0, column=1, padx=10, pady=10)
        
        # Панель информации
        info_frame = ttk.Frame(self.root, padding="10")
        info_frame.grid(row=2, column=0, sticky=(tk.W, tk.E))
        
        self.info_label = ttk.Label(
            info_frame,
            text="Выберите режим работы",
            font=('Helvetica', 12)
        )
        self.info_label.grid(row=0, column=0, pady=5)
        
        # Координаты взгляда
        self.coords_label = ttk.Label(
            info_frame,
            text="Координаты взгляда (нормализованные): (0.00, 0.00)",
            font=('Helvetica', 10)
        )
        self.coords_label.grid(row=1, column=0, pady=5)
        
        # Координаты в градусах
        self.degrees_label = ttk.Label(
            info_frame,
            text="Координаты взгляда (градусы): (0.0°, 0.0°)",
            font=('Helvetica', 10)
        )
        self.degrees_label.grid(row=2, column=0, pady=5)
        
        # Настройка весов
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)
        self.display_frame.columnconfigure(0, weight=1)
        self.display_frame.rowconfigure(0, weight=1)
    
    def show_info(self):
        """Показать информацию о системе"""
        info_text = """
        Columbia Gaze Tracking System
        Версия 1.0
        
        Модель обучена на Columbia Gaze Dataset:
        - Горизонтальный взгляд: от -15° до +15°
        - Вертикальный взгляд: от -10° до +20°
        
        Красная точка показывает направление взгляда
        на виртуальном мониторе.
        """
        messagebox.showinfo("Информация о системе", info_text)
    
    def load_image_mode(self):
        """Режим загрузки фото"""
        self.stop_camera()
        
        file_path = filedialog.askopenfilename(
            title="Выберите изображение",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        
        if file_path:
            self.process_image_file(file_path)
    
    def process_image_file(self, file_path):
        """Обработка загруженного изображения"""
        try:
            # Загрузка изображения
            image = cv2.imread(file_path)
            if image is None:
                raise ValueError("Не удалось загрузить изображение")
                
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.current_image = image.copy()
            
            # Отображение изображения
            self.display_image(image)
            
            # Определение направления взгляда
            gaze_normalized, gaze_degrees = self.predict_gaze(image)
            
            # Визуализация результатов
            self.visualize_gaze(gaze_normalized, gaze_degrees)
            
            self.info_label.config(text="Обработка фото завершена")
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось обработать изображение: {str(e)}")
    
    def camera_mode(self):
        """Режим работы с камерой"""
        if self.camera_active:
            return
        
        self.camera_active = True
        self.cap = cv2.VideoCapture(0)
        
        if not self.cap.isOpened():
            messagebox.showerror("Ошибка", "Не удалось открыть камеру")
            self.camera_active = False
            return
        
        self.info_label.config(text="Камера запущена - смотрите прямо в камеру")
        
        # Запуск потока обработки кадров
        self.process_camera_frames()
    
    def process_camera_frames(self):
        """Обработка кадров с камеры"""
        if self.camera_active and self.cap.isOpened():
            ret, frame = self.cap.read()
            
            if ret:
                # Конвертация цвета
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.current_image = frame_rgb.copy()
                
                # Отображение кадра
                self.display_image(frame_rgb)
                
                # Определение направления взгляда
                gaze_normalized, gaze_degrees = self.predict_gaze(frame_rgb)
                
                # Визуализация результатов
                self.visualize_gaze(gaze_normalized, gaze_degrees)
            
            # Планирование следующего кадра
            if self.camera_active:
                self.root.after(30, self.process_camera_frames)  # ~30 FPS
    
    def stop_camera(self):
        """Остановка камеры"""
        self.camera_active = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.info_label.config(text="Камера остановлена")
    
    def display_image(self, image):
        """Отображение изображения в интерфейсе"""
        # Изменение размера для отображения
        display_img = cv2.resize(image, (400, 300))
        
        # Конвертация для tkinter
        img_pil = Image.fromarray(display_img)
        img_tk = ImageTk.PhotoImage(img_pil)
        
        self.image_label.config(image=img_tk)
        self.image_label.image = img_tk
    
    def predict_gaze(self, image):
        """Предсказание направления взгляда"""
        try:
            # Обнаружение лица и вырезка
            face_region = self.extract_face_region(image)
            
            # Предобработка изображения
            img_pil = Image.fromarray(face_region)
            img_tensor = self.transform(img_pil).unsqueeze(0).to(self.device)
            
            # Предсказание
            with torch.no_grad():
                prediction = self.model(img_tensor)
                gaze_normalized = prediction.cpu().numpy()[0]
            
            # Денормализация в градусы
            gaze_degrees = np.array([
                gaze_normalized[0] * 15,  # горизонтальный
                gaze_normalized[1] * 20   # вертикальный
            ])
            
            # Обновление информации
            self.coords_label.config(
                text=f"Координаты взгляда (нормализованные): ({gaze_normalized[0]:.2f}, {gaze_normalized[1]:.2f})"
            )
            
            self.degrees_label.config(
                text=f"Координаты взгляда (градусы): ({gaze_degrees[0]:.1f}°, {gaze_degrees[1]:.1f}°)"
            )
            
            return gaze_normalized, gaze_degrees
            
        except Exception as e:
            print(f"Ошибка предсказания: {e}")
            return np.array([0.0, 0.0]), np.array([0.0, 0.0])
    
    def extract_face_region(self, image):
        """Вырезка области лица из изображения"""
        # Для простоты вырезаем центральную часть
        height, width = image.shape[:2]
        
        # Размер области лица (предполагаем лицо в центре)
        face_size = min(height, width) // 2
        
        # Координаты центра
        center_x = width // 2
        center_y = height // 2
        
        # Вырезаем квадратную область
        x1 = max(0, center_x - face_size // 2)
        y1 = max(0, center_y - face_size // 2)
        x2 = min(width, center_x + face_size // 2)
        y2 = min(height, center_y + face_size // 2)
        
        face_region = image[y1:y2, x1:x2]
        
        # Ресайз до 224x224
        face_region = cv2.resize(face_region, (224, 224))
        
        return face_region
    
    def visualize_gaze(self, gaze_normalized, gaze_degrees):
        """Визуализация направления взгляда на экране"""
        # Очистка canvas
        self.gaze_canvas.delete("all")
        
        # Рисование монитора
        monitor_width = 700
        monitor_height = 400
        monitor_x = 50
        monitor_y = 100
        
        # Монитор
        self.gaze_canvas.create_rectangle(
            monitor_x, monitor_y,
            monitor_x + monitor_width,
            monitor_y + monitor_height,
            fill="black", outline="white", width=3
        )
        
        # Разметка монитора
        for i in range(1, 4):
            x_pos = monitor_x + (monitor_width // 4) * i
            self.gaze_canvas.create_line(
                x_pos, monitor_y,
                x_pos, monitor_y + monitor_height,
                fill="gray", width=1, dash=(2, 2)
            )
        
        for i in range(1, 3):
            y_pos = monitor_y + (monitor_height // 3) * i
            self.gaze_canvas.create_line(
                monitor_x, y_pos,
                monitor_x + monitor_width, y_pos,
                fill="gray", width=1, dash=(2, 2)
            )
        
        # Преобразование нормализованных координат в координаты на мониторе
        # Нормализованные: [-1, 1] -> Экранные: [0, monitor_width/height]
        screen_x = monitor_x + (gaze_normalized[0] + 1) / 2 * monitor_width
        screen_y = monitor_y + (gaze_normalized[1] + 1) / 2 * monitor_height
        
        # Ограничиваем точку на экране
        screen_x = max(monitor_x, min(monitor_x + monitor_width, screen_x))
        screen_y = max(monitor_y, min(monitor_y + monitor_height, screen_y))
        
        # Рисование точки взгляда
        point_radius = 12
        self.gaze_canvas.create_oval(
            screen_x - point_radius, screen_y - point_radius,
            screen_x + point_radius, screen_y + point_radius,
            fill="red", outline="yellow", width=3
        )
        
        # Добавление текста с координатами
        self.gaze_canvas.create_text(
            screen_x, screen_y - 25,
            text=f"H: {gaze_degrees[0]:.1f}°, V: {gaze_degrees[1]:.1f}°",
            fill="white", font=("Arial", 10, "bold")
        )
        
        # Рисование перекрестия
        self.gaze_canvas.create_line(
            screen_x, monitor_y,
            screen_x, monitor_y + monitor_height,
            fill="red", width=1, dash=(4, 2)
        )
        self.gaze_canvas.create_line(
            monitor_x, screen_y,
            monitor_x + monitor_width, screen_y,
            fill="red", width=1, dash=(4, 2)
        )
        
        # Заголовок
        self.gaze_canvas.create_text(
            monitor_x + monitor_width // 2, 50,
            text="Визуализация направления взгляда на мониторе",
            fill="black", font=("Arial", 14, "bold")
        )
        
        # Легенда
        self.gaze_canvas.create_text(
            monitor_x + monitor_width // 2, monitor_y + monitor_height + 30,
            text="Красная точка показывает, куда смотрит пользователь",
            fill="black", font=("Arial", 10)
        )
    
    def on_closing(self):
        """Обработка закрытия приложения"""
        self.stop_camera()
        self.root.destroy()

# Главная функция
def main():
    root = tk.Tk()
    app = ColumbiaGazeApp(root)
    
    # Обработка закрытия окна
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    
    root.mainloop()

if __name__ == "__main__":
    main()