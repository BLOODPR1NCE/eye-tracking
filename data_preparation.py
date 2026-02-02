# data_preparation.py
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import gc

class MPIIGazeDataLoader:
   
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.participants = [f'p{str(i).zfill(2)}' for i in range(15)]
        
    def load_all_original_data(self):
        print("Загрузка ВСЕХ данных из MPIIGaze...")
        
        all_left_eyes = []
        all_right_eyes = []
        all_targets_x = []
        all_targets_y = []
        total_images_processed = 0
        total_images_skipped = 0
        
        for participant in self.participants[:3]:
            participant_path = os.path.join(self.dataset_path, 'Data', 'Original', participant)
            
            if not os.path.exists(participant_path):
                continue
            
            days = [d for d in os.listdir(participant_path) 
                   if os.path.isdir(os.path.join(participant_path, d)) and d.lower() != 'calibration']
            
            print(f"\nУчастник {participant}: {len(days)} дней")
            
            for day in tqdm(days, desc=f"  Дни {participant}", leave=False):
                day_path = os.path.join(participant_path, day)
                
                annotation_file = os.path.join(day_path, 'annotation.txt')
                if not os.path.exists(annotation_file):
                    continue
                
                try:
                    annotations = np.loadtxt(annotation_file)
                except:
                    continue
                
                if len(annotations.shape) == 1:
                    annotations = annotations.reshape(1, -1)
                
                image_files = []
                for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                    image_files.extend([f for f in os.listdir(day_path) 
                                      if f.lower().endswith(ext)])
                
                if not image_files:
                    continue
                
                print(f"    День {day}: {len(image_files)} изображений, {annotations.shape[0]} аннотаций")
                
                # Обрабатываем каждое изображение
                processed_in_day = 0
                skipped_in_day = 0
                
                for img_idx, img_file in enumerate(tqdm(image_files, desc=f"      Изображения", leave=False)):
                    if img_idx >= len(annotations):
                        skipped_in_day += 1
                        continue
                    
                    img_path = os.path.join(day_path, img_file)
                    
                    img = cv2.imread(img_path)
                    if img is None:
                        skipped_in_day += 1
                        continue
                    
                    ann = annotations[img_idx]
                    
                    if ann.shape[0] < 26:
                        print(f"      Предупреждение: аннотация {img_idx} имеет неправильный размер: {ann.shape}")
                        skipped_in_day += 1
                        continue
                    
                    try:
                        screen_x = ann[24] / 1920.0  # Индекс 24 = 25-й элемент 
                        screen_y = ann[25] / 1080.0  # Индекс 25 = 26-й элемент
                        
                        if not (0 <= screen_x <= 1 and 0 <= screen_y <= 1):
                            print(f"      Предупреждение: некорректные координаты [{screen_x}, {screen_y}]")
                            skipped_in_day += 1
                            continue
                        
                    except Exception as e:
                        print(f"      Ошибка извлечения координат: {e}")
                        skipped_in_day += 1
                        continue
                    
                    left_eye, right_eye = self.extract_eyes_from_image(img)
                    
                    if left_eye is None or right_eye is None:
                        skipped_in_day += 1
                        continue
                    
                    try:
                        left_eye = cv2.resize(left_eye, (36, 36))
                        right_eye = cv2.resize(right_eye, (36, 36))
                        left_eye = cv2.equalizeHist(left_eye)
                        right_eye = cv2.equalizeHist(right_eye)

                        all_left_eyes.append(left_eye.astype('float32') / 255.0)
                        all_right_eyes.append(right_eye.astype('float32') / 255.0)
                        all_targets_x.append(screen_x)
                        all_targets_y.append(screen_y)
                        
                        processed_in_day += 1
                        total_images_processed += 1
                        
                    except Exception as e:
                        print(f"      Ошибка обработки глаз: {e}")
                        skipped_in_day += 1
                        continue
                    
                    if total_images_processed % 1000 == 0:
                        gc.collect()
                
                print(f"    Обработано: {processed_in_day} изображений, пропущено: {skipped_in_day}")
                total_images_skipped += skipped_in_day
            
            print(f"\n  Участник {participant} завершен:")
            print(f"    Обработано: {total_images_processed} изображений")
            print(f"    Пропущено: {total_images_skipped} изображений")
        
        print(f"\nИтоговая статистика:")
        print(f"  Всего обработано изображений: {total_images_processed}")
        print(f"  Пропущено изображений: {total_images_skipped}")
        
        if total_images_processed == 0:
            return None, None, None
        
        print("\nПреобразование данных в numpy массивы...")
        try:
            left_X = np.array(all_left_eyes).reshape(-1, 36, 36, 1)
            right_X = np.array(all_right_eyes).reshape(-1, 36, 36, 1)
            targets = np.column_stack([all_targets_x, all_targets_y])
            
            print(f"\nСоздан датасет:")
            
            return left_X, right_X, targets
            
        except Exception as e:
            print(f"Ошибка при создании numpy массивов: {e}")
            return None, None, None
    
    def extract_eyes_from_image(self, img):
        try:
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img
            
            gray = cv2.equalizeHist(gray)
            
            if not hasattr(self, 'face_cascade'):
                self.face_cascade = cv2.CascadeClassifier(
                    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                )
                self.eye_cascade = cv2.CascadeClassifier(
                    cv2.data.haarcascades + 'haarcascade_eye.xml'
                )
            
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,minSize=(100, 100))
            
            if len(faces) == 0:
                eyes = self.eye_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=3,minSize=(30, 30))
                
                if len(eyes) >= 2:
                    eyes = sorted(eyes, key=lambda e: e[0])
                    ex1, ey1, ew1, eh1 = eyes[0]
                    ex2, ey2, ew2, eh2 = eyes[1]
                    
                    left_eye = gray[ey1:ey1+eh1, ex1:ex1+ew1]
                    right_eye = gray[ey2:ey2+eh2, ex2:ex2+ew2]
                    
                    return left_eye, right_eye
                return None, None
            
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            face_roi = gray[y:y+h, x:x+w]
            
            eyes = self.eye_cascade.detectMultiScale(face_roi,scaleFactor=1.1,minNeighbors=3,minSize=(30, 30))
            
            if len(eyes) < 2:
                return None, None
            
            eyes = sorted(eyes, key=lambda e: e[0])
            
            ex1, ey1, ew1, eh1 = eyes[0]
            ex2, ey2, ew2, eh2 = eyes[1]
            
            left_eye = face_roi[ey1:ey1+eh1, ex1:ex1+ew1]
            right_eye = face_roi[ey2:ey2+eh2, ex2:ex2+ew2]
            
            return left_eye, right_eye
            
        except Exception as e:
            print(f"Ошибка при извлечении глаз: {e}")
            return None, None

def prepare_full_mpiigaze_dataset(dataset_path):
    print("=" * 60)
    print("ПОДГОТОВКА ПОЛНОГО ДАТАСЕТА MPIIGAZE")
    print("=" * 60)
    
    loader = MPIIGazeDataLoader(dataset_path)
    
    print("\nЗагрузка данных из MPIIGaze...")
    left_X, right_X, y = loader.load_all_original_data()
    
    if left_X is None or len(left_X) == 0:
        print("\nНе удалось загрузить данные.")
    
    if left_X is not None and len(left_X) > 0:
        print(f"\nУспешно загружено {len(left_X)} пар глаз")
        
        print(f"\nСтатистика датасета:")
        print(f"  Размер: {len(left_X)} samples")
        print(f"  Форма изображений: {left_X.shape[1:]}")
        print(f"  Диапазон X: {y[:, 0].min():.4f} - {y[:, 0].max():.4f}")
        print(f"  Диапазон Y: {y[:, 1].min():.4f} - {y[:, 1].max():.4f}")
        print(f"  Среднее X: {y[:, 0].mean():.4f}")
        print(f"  Среднее Y: {y[:, 1].mean():.4f}")
        
        return left_X, right_X, y
    
    return None, None, None

def save_datasets(left_X, right_X, y, output_prefix='mpiigaze_full'):
    from sklearn.model_selection import train_test_split
    
    print("\nРазделение и сохранение данных...")
    
    output_dir = 'processed_data'
    os.makedirs(output_dir, exist_ok=True)
    
    # Разделяем индексы
    indices = np.arange(len(left_X))
    
    # 70% train, 15% validation, 15% test
    train_idx, temp_idx = train_test_split(indices, test_size=0.3, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)
    
    print(f"Разделение данных:")
    print(f"  Train: {len(train_idx)} samples ({len(train_idx)/len(indices)*100:.1f}%)")
    print(f"  Validation: {len(val_idx)} samples ({len(val_idx)/len(indices)*100:.1f}%)")
    print(f"  Test: {len(test_idx)} samples ({len(test_idx)/len(indices)*100:.1f}%)")
    
    np.save(os.path.join(output_dir, f'{output_prefix}_left_train.npy'), left_X[train_idx])
    np.save(os.path.join(output_dir, f'{output_prefix}_right_train.npy'), right_X[train_idx])
    np.save(os.path.join(output_dir, f'{output_prefix}_targets_train.npy'), y[train_idx])
    
    np.save(os.path.join(output_dir, f'{output_prefix}_left_val.npy'), left_X[val_idx])
    np.save(os.path.join(output_dir, f'{output_prefix}_right_val.npy'), right_X[val_idx])
    np.save(os.path.join(output_dir, f'{output_prefix}_targets_val.npy'), y[val_idx])
    
    np.save(os.path.join(output_dir, f'{output_prefix}_left_test.npy'), left_X[test_idx])
    np.save(os.path.join(output_dir, f'{output_prefix}_right_test.npy'), right_X[test_idx])
    np.save(os.path.join(output_dir, f'{output_prefix}_targets_test.npy'), y[test_idx])
    
    # Сохраняем индексы для воспроизводимости
    np.save(os.path.join(output_dir, f'{output_prefix}_train_idx.npy'), train_idx)
    np.save(os.path.join(output_dir, f'{output_prefix}_val_idx.npy'), val_idx)
    np.save(os.path.join(output_dir, f'{output_prefix}_test_idx.npy'), test_idx)
    
    stats = {
        'total_samples': len(left_X),
        'image_shape': left_X.shape[1:],
        'train_samples': len(train_idx),
        'val_samples': len(val_idx),
        'test_samples': len(test_idx),
        'targets_min': y.min(axis=0).tolist(),
        'targets_max': y.max(axis=0).tolist(),
        'targets_mean': y.mean(axis=0).tolist(),
        'targets_std': y.std(axis=0).tolist()
    }
    
    import json
    with open(os.path.join(output_dir, f'{output_prefix}_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nДанные сохранены в папке '{output_dir}':")
    for file in os.listdir(output_dir):
        if file.startswith(output_prefix):
            print(f"  {file}")
    
    return output_dir

def visualize_full_dataset(left_X, right_X, y, output_dir):
    print("\nСоздание визуализации датасета...")
    
    # Ограничиваем количество для визуализации
    n_samples = min(10000, len(left_X))
    indices = np.random.choice(len(left_X), n_samples, replace=False)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Распределение целей
    axes[0, 0].scatter(y[indices, 0], y[indices, 1], alpha=0.1, s=5, c='blue')
    axes[0, 0].set_xlabel('X (нормализовано)', fontsize=12)
    axes[0, 0].set_ylabel('Y (нормализовано)', fontsize=12)
    axes[0, 0].set_title('Распределение целей взгляда', fontsize=14)
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Гистограмма X
    axes[0, 1].hist(y[:, 0], bins=50, edgecolor='black', alpha=0.7, color='skyblue')
    axes[0, 1].set_xlabel('X координата', fontsize=12)
    axes[0, 1].set_ylabel('Частота', fontsize=12)
    axes[0, 1].set_title('Распределение X координат', fontsize=14)
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Гистограмма Y
    axes[0, 2].hist(y[:, 1], bins=50, edgecolor='black', alpha=0.7, color='lightcoral')
    axes[0, 2].set_xlabel('Y координата', fontsize=12)
    axes[0, 2].set_ylabel('Частота', fontsize=12)
    axes[0, 2].set_title('Распределение Y координат', fontsize=14)
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Примеры глаз
    for i in range(6):
        row = 1 if i < 3 else 1
        col = i % 3
        
        if i % 2 == 0:
            # Левые глаза
            img = left_X[i].squeeze()
            title = f'Левый глаз {i//2+1}'
            target = y[i]
        else:
            # Правые глаза
            img = right_X[i-1].squeeze()
            title = f'Правый глаз {i//2+1}'
            target = y[i-1]
        
        axes[row, col].imshow(img, cmap='gray', vmin=0, vmax=1)
        axes[row, col].set_title(f'{title}\nX={target[0]:.2f}, Y={target[1]:.2f}', fontsize=10)
        axes[row, col].axis('off')
    
    plt.suptitle(f'Анализ датасета MPIIGaze ({len(left_X)} samples)', fontsize=16, y=1.02)
    plt.tight_layout()
    
    # Сохраняем визуализацию
    output_path = os.path.join(output_dir, 'dataset_analysis.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Визуализация сохранена: {output_path}")

if __name__ == "__main__":
    print("=" * 60)
    print("ПОДГОТОВКА ПОЛНОГО ДАТАСЕТА MPIIGAZE")
    print("=" * 60)
    
    mpiigaze_path = r"C:\Users\prince\Downloads\MPIIGaze\MPIIGaze"
    
    if not os.path.exists(mpiigaze_path):
        print(f"Путь не существует: {mpiigaze_path}")
    else:    
        left_X, right_X, y = prepare_full_mpiigaze_dataset(mpiigaze_path)
        
        if left_X is None:
            print("\nНе удалось загрузить MPIIGaze.")
    
    if left_X is not None and len(left_X) > 0:
        output_dir = save_datasets(left_X, right_X, y, output_prefix='mpiigaze_full')
        
        visualize_full_dataset(left_X, right_X, y, output_dir)
        
        print("\n" + "=" * 60)
        print("ПОДГОТОВКА ДАННЫХ ЗАВЕРШЕНА!")
        print("=" * 60)
        print(f"\nСоздан датасет из {len(left_X)} пар глаз")
        print(f"Данные сохранены в папке: {output_dir}")
    else:
        print("\nНе удалось создать датасет. Проверьте путь к данным.")