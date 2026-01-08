# columbia_data_preparation.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class ColumbiaDataPreprocessor:
    def __init__(self, base_path):
        self.base_path = base_path
        self.data = []
        self.labels = []
        
    def parse_filename(self, filename):
        """Парсинг имени файла для извлечения меток"""
        # Пример: 0011_2m_0P_10V_-5H.jpg
        parts = filename.replace('.jpg', '').split('_')
        
        if len(parts) >= 5:
            person_id = parts[0]
            distance = parts[1]  # 2m
            head_pose = parts[2]  # 0P
            gaze_v = parts[3]    # 10V
            gaze_h = parts[4]    # -5H
            
            # Извлекаем числовые значения
            try:
                head_angle = int(head_pose.replace('P', ''))
                gaze_vertical = int(gaze_v.replace('V', ''))
                gaze_horizontal = int(gaze_h.replace('H', ''))
                
                return {
                    'person_id': person_id,
                    'distance': distance,
                    'head_pose': head_angle,
                    'gaze_v': gaze_vertical,
                    'gaze_h': gaze_horizontal,
                    'gaze_v_rad': np.deg2rad(gaze_vertical),
                    'gaze_h_rad': np.deg2rad(gaze_horizontal)
                }
            except:
                return None
        return None
    
    def load_dataset_info(self):
        """Загрузка информации о датасете"""
        print("Загрузка Columbia Gaze Data Set...")
        
        all_images = []
        for person_dir in tqdm(sorted(os.listdir(self.base_path))):
            person_path = os.path.join(self.base_path, person_dir)
            
            if os.path.isdir(person_path):
                for img_file in sorted(os.listdir(person_path)):
                    if img_file.lower().endswith('.jpg'):
                        img_path = os.path.join(person_path, img_file)
                        
                        # Парсинг меток из имени файла
                        metadata = self.parse_filename(img_file)
                        
                        if metadata:
                            all_images.append({
                                'person': person_dir,
                                'image_name': img_file,
                                'image_path': img_path,
                                'head_pose': metadata['head_pose'],
                                'gaze_v': metadata['gaze_v'],
                                'gaze_h': metadata['gaze_h'],
                                'gaze_v_rad': metadata['gaze_v_rad'],
                                'gaze_h_rad': metadata['gaze_h_rad']
                            })
        
        self.data = all_images
        print(f"Найдено {len(self.data)} изображений")
        return self.data
    
    def analyze_correlations(self):
        """Анализ корреляций"""
        if not self.data:
            print("Данные не загружены")
            return
        
        # Создаем DataFrame для анализа
        analysis_data = []
        for item in self.data:
            analysis_data.append({
                'gaze_h': item['gaze_h'],
                'gaze_v': item['gaze_v'],
                'head_pose': item['head_pose']
            })
        
        df = pd.DataFrame(analysis_data)
        
        # Матрица корреляций
        correlation_matrix = df.corr()
        
        print("Матрица корреляций:")
        print(correlation_matrix)
        
        # Визуализация корреляций
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Матрица корреляций переменных - Columbia Dataset')
        plt.tight_layout()
        plt.savefig('columbia_correlation_matrix.png')
        plt.show()
        
        return correlation_matrix
    
    def plot_distributions(self):
        """Построение графиков распределений"""
        if not self.data:
            print("Данные не загружены")
            return
        
        # Создаем данные для гистограмм
        gaze_h = [item['gaze_h'] for item in self.data]
        gaze_v = [item['gaze_v'] for item in self.data]
        head_pose = [item['head_pose'] for item in self.data]
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Распределение горизонтального взгляда
        axes[0].hist(gaze_h, bins=30, edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('Horizontal Gaze (degrees)')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Distribution of Horizontal Gaze')
        axes[0].grid(True, alpha=0.3)
        
        # Распределение вертикального взгляда
        axes[1].hist(gaze_v, bins=30, edgecolor='black', alpha=0.7, color='orange')
        axes[1].set_xlabel('Vertical Gaze (degrees)')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Distribution of Vertical Gaze')
        axes[1].grid(True, alpha=0.3)
        
        # Распределение позы головы
        axes[2].hist(head_pose, bins=10, edgecolor='black', alpha=0.7, color='green')
        axes[2].set_xlabel('Head Pose (degrees)')
        axes[2].set_ylabel('Frequency')
        axes[2].set_title('Distribution of Head Pose')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('columbia_gaze_distributions.png')
        plt.show()
        
        # Scatter plot
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(gaze_h, gaze_v, c=head_pose, cmap='viridis', 
                           alpha=0.6, s=20)
        ax.set_xlabel('Horizontal Gaze (degrees)')
        ax.set_ylabel('Vertical Gaze (degrees)')
        ax.set_title('Scatter Plot of Gaze Directions')
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, label='Head Pose (degrees)')
        plt.savefig('columbia_gaze_scatter.png')
        plt.show()
        
        # Q-Q plots для проверки нормальности
        from scipy import stats
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        stats.probplot(gaze_h, dist="norm", plot=axes[0])
        axes[0].set_title('Q-Q plot для Horizontal Gaze')
        
        stats.probplot(gaze_v, dist="norm", plot=axes[1])
        axes[1].set_title('Q-Q plot для Vertical Gaze')
        
        plt.tight_layout()
        plt.savefig('columbia_qq_plots.png')
        plt.show()
    
    def prepare_training_data(self, output_dir='columbia_processed'):
        """Подготовка данных для обучения"""
        os.makedirs(output_dir, exist_ok=True)
        
        print("Подготовка данных для обучения...")
        
        # Создаем структуру для обучения
        train_data = []
        
        for item in tqdm(self.data[:2000]):  # Ограничиваем для примера
            try:
                # Загрузка изображения
                img = Image.open(item['image_path'])
                img = img.convert('RGB')
                
                # Обнаружение лица (поскольку лица в центре, можно просто вырезать центр)
                width, height = img.size
                face_size = min(width, height) // 2
                left = (width - face_size) // 2
                top = (height - face_size) // 2
                right = left + face_size
                bottom = top + face_size
                
                # Вырезаем лицо из центра
                face_img = img.crop((left, top, right, bottom))
                face_img = face_img.resize((224, 224))
                
                # Сохранение обработанных данных
                img_filename = f"{item['person']}_{item['image_name'].replace('.jpg', '')}.jpg"
                img.save(os.path.join(output_dir, img_filename))
                
                # Нормализация меток для нейросети (в диапазон [-1, 1])
                # Диапазоны из датасета Columbia:
                # Горизонтальный: примерно -15 до +15 градусов
                # Вертикальный: примерно -10 до +20 градусов
                gaze_h_norm = item['gaze_h'] / 15.0  # нормализуем к [-1, 1]
                gaze_v_norm = item['gaze_v'] / 20.0  # нормализуем к [-1, 1]
                
                # Ограничиваем значения на случай выбросов
                gaze_h_norm = np.clip(gaze_h_norm, -1, 1)
                gaze_v_norm = np.clip(gaze_v_norm, -1, 1)
                
                train_data.append({
                    'image': img_filename,
                    'gaze_x': gaze_h_norm,  # горизонтальный взгляд
                    'gaze_y': gaze_v_norm,  # вертикальный взгляд
                    'gaze_h_original': item['gaze_h'],
                    'gaze_v_original': item['gaze_v'],
                    'head_pose': item['head_pose']
                })
            except Exception as e:
                print(f"Ошибка при обработке {item['image_path']}: {e}")
        
        # Сохранение метаданных
        metadata_df = pd.DataFrame(train_data)
        metadata_df.to_csv(os.path.join(output_dir, 'metadata.csv'), index=False)
        
        print(f"Подготовлено {len(train_data)} примеров")
        return output_dir

# Использование
if __name__ == "__main__":
    base_path = r"C:\Users\prince\Downloads\Columbia Gaze Data Set"
    
    preprocessor = ColumbiaDataPreprocessor(base_path)
    preprocessor.load_dataset_info()
    
    # Анализ
    preprocessor.analyze_correlations()
    preprocessor.plot_distributions()
    
    # Подготовка данных
    processed_dir = preprocessor.prepare_training_data()