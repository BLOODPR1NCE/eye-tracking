#data_preparation.py
import os
import re
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def parse_filename_simple(filename):
    """Простое преобразование координат"""
    pattern = r'(\d+)_(\d+m)_([-\d]+)P_(\d+)V_([-\d]+)H\.jpg'
    match = re.search(pattern, filename)
    if match:
        vertical = int(match.group(4))    # V: 0, 10, 20
        horizontal = int(match.group(5))  # H: -15...15
        
        # Преобразуем в координаты 0-1
        screen_x = (horizontal + 15) / 30.0  # -15°->0.0, 0°->0.5, 15°->1.0
        screen_y = vertical / 20.0           # 0°->0.0, 20°->1.0
        
        return screen_x, screen_y
    return None, None

def extract_face_eyes_simple(img):
    """Извлекаем лицо и глаза из изображения"""
    # Конвертируем в grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Загружаем каскады
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    eye_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_eye.xml'
    )
    
    # Находим лицо
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))
    
    if len(faces) == 0:
        return None, None, None
    
    # Берем самое большое лицо
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    face_roi = gray[y:y+h, x:x+w]
    
    # Находим глаза
    eyes = eye_cascade.detectMultiScale(face_roi, 1.1, 3, minSize=(20, 20))
    
    if len(eyes) < 2:
        return face_roi, None, None
    
    # Сортируем глаза по X координате
    eyes = sorted(eyes, key=lambda e: e[0])
    
    # Извлекаем левый глаз (первый в списке)
    ex1, ey1, ew1, eh1 = eyes[0]
    left_eye = face_roi[ey1:ey1+eh1, ex1:ex1+ew1]
    
    # Извлекаем правый глаз (второй в списке)
    ex2, ey2, ew2, eh2 = eyes[1]
    right_eye = face_roi[ey2:ey2+eh2, ex2:ex2+ew2]
    
    return face_roi, left_eye, right_eye

def prepare_simple_dataset(data_path, max_samples=1000):
    """Подготовка упрощенного датасета"""
    print(f"Загрузка данных из {data_path}")
    
    left_eyes = []
    right_eyes = []
    targets = []
    faces = []
    
    folders = [f for f in os.listdir(data_path) 
               if os.path.isdir(os.path.join(data_path, f))]
    folders.sort()
    
    # Берем первые 10 папок для скорости
    folders = folders[:56]
    
    for folder in tqdm(folders, desc="Обработка"):
        folder_path = os.path.join(data_path, folder)
        
        for filename in os.listdir(folder_path):
            if filename.endswith('.jpg') and len(left_eyes) < max_samples:
                x, y = parse_filename_simple(filename)
                if x is None:
                    continue
                
                img_path = os.path.join(folder_path, filename)
                img = cv2.imread(img_path)
                
                if img is not None:
                    # Извлекаем лицо и глаза
                    face_roi, left_eye, right_eye = extract_face_eyes_simple(img)
                    
                    if left_eye is not None and right_eye is not None:
                        # Ресайз глаз до 32x32
                        left_resized = cv2.resize(left_eye, (32, 32))
                        right_resized = cv2.resize(right_eye, (32, 32))
                        
                        # Улучшение контраста
                        left_eq = cv2.equalizeHist(left_resized)
                        right_eq = cv2.equalizeHist(right_resized)
                        
                        # Нормализация
                        left_norm = left_eq.astype('float32') / 255.0
                        right_norm = right_eq.astype('float32') / 255.0
                        
                        left_eyes.append(left_norm)
                        right_eyes.append(right_norm)
                        targets.append([x, y])
                        
                        if face_roi is not None:
                            face_resized = cv2.resize(face_roi, (64, 64))
                            face_eq = cv2.equalizeHist(face_resized)
                            face_norm = face_eq.astype('float32') / 255.0
                            faces.append(face_norm)
    
    # Преобразуем в numpy
    left_X = np.array(left_eyes).reshape(-1, 32, 32, 1)
    right_X = np.array(right_eyes).reshape(-1, 32, 32, 1)
    y = np.array(targets)
    
    print(f"\nСоздано {len(left_X)} пар глаз")
    print(f"Диапазон X: {y[:, 0].min():.3f} - {y[:, 0].max():.3f}")
    print(f"Диапазон Y: {y[:, 1].min():.3f} - {y[:, 1].max():.3f}")
    
    return left_X, right_X, y

# Запуск
if __name__ == "__main__":
    data_path = r"C:\Users\prince\Downloads\Columbia Gaze Data Set"
    
    # Подготовка данных
    left_X, right_X, y = prepare_simple_dataset(data_path, max_samples=5000)
    
    # Сохранение данных
    np.save('left_eyes_simple.npy', left_X)
    np.save('right_eyes_simple.npy', right_X)
    np.save('targets_simple.npy', y)
    
    print("\nДанные сохранены!")
    print("left_eyes_simple.npy")
    print("right_eyes_simple.npy")
    print("targets_simple.npy")
    
    # Визуализация
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    
    axes[0, 0].scatter(y[:, 0], y[:, 1], alpha=0.5, s=10)
    axes[0, 0].set_title('Распределение целей')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].hist(y[:, 0], bins=20, edgecolor='black')
    axes[0, 1].set_title('Распределение X')
    
    axes[0, 2].hist(y[:, 1], bins=20, edgecolor='black')
    axes[0, 2].set_title('Распределение Y')
    
    # Примеры глаз
    for i in range(3):
        axes[1, i].imshow(left_X[i].squeeze(), cmap='gray')
        axes[1, i].set_title(f'Пример {i+1}')
        axes[1, i].axis('off')
    
    axes[1, 3].imshow(right_X[0].squeeze(), cmap='gray')
    axes[1, 3].set_title('Правый глаз')
    axes[1, 3].axis('off')  
    
    plt.tight_layout()
    plt.savefig('simple_data_analysis.png', dpi=100)
    plt.show()