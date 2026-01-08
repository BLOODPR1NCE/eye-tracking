# columbia_model_training.py
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import pandas as pd
from PIL import Image
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class ColumbiaGazeDataset(Dataset):
    """Датасет для Columbia Gaze Data Set с реальной загрузкой изображений"""
    def __init__(self, metadata_df, images_dir, transform=None):
        self.metadata = metadata_df
        self.images_dir = images_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        img_filename = self.metadata.iloc[idx]['image']
        gaze_x = self.metadata.iloc[idx]['gaze_x']
        gaze_y = self.metadata.iloc[idx]['gaze_y']
        
        img_path = os.path.join(self.images_dir, img_filename)
        
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            # Если файл не найден, создаем черное изображение
            print(f"Предупреждение: файл {img_path} не найден")
            image = Image.new('RGB', (224, 224), color='black')
        
        if self.transform:
            image = self.transform(image)
        
        # Нормализованные метки (уже в диапазоне [-1, 1])
        gaze = torch.tensor([gaze_x, gaze_y], dtype=torch.float32)
        
        # Оригинальные метки в градусах (для оценки)
        gaze_h_original = torch.tensor([self.metadata.iloc[idx]['gaze_h_original']], dtype=torch.float32)
        gaze_v_original = torch.tensor([self.metadata.iloc[idx]['gaze_v_original']], dtype=torch.float32)
        
        return image, gaze, gaze_h_original, gaze_v_original

class EfficientGazeNet(nn.Module):
    """Эффективная модель для определения направления взгляда"""
    def __init__(self):
        super(EfficientGazeNet, self).__init__()
        
        # Эффективная сверточная архитектура
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        
        # Глобальный average pooling вместо полносвязных слоев
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Регрессионная голова
        self.regressor = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 2),
            nn.Tanh()  # Ограничиваем выход в диапазоне [-1, 1]
        )
        
    def forward(self, x):
        x = self.conv_block1(x)    # 112x112
        x = self.conv_block2(x)    # 56x56
        x = self.conv_block3(x)    # 28x28
        x = self.conv_block4(x)    # 14x14
        
        # Глобальный пулинг
        x = self.global_pool(x)    # 1x1
        x = x.view(x.size(0), -1)  # flatten
        
        # Регрессия
        x = self.regressor(x)
        
        return x

class ColumbiaGazeTrainer:
    def __init__(self, data_dir, metadata_file):
        self.data_dir = data_dir
        self.metadata_file = metadata_file
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Используется устройство: {self.device}")
        
        # Инициализация модели
        self.model = EfficientGazeNet().to(self.device)
        
        # Функция потерь и оптимизатор
        self.criterion = nn.MSELoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=0.01)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=10)
        
        # Счетчики для обучения
        self.train_losses = []
        self.val_losses = []
        self.val_errors = []
        
    def prepare_datasets(self):
        """Подготовка train/validation/test наборов"""
        print("Подготовка данных для обучения...")
        
        # Загрузка метаданных
        metadata = pd.read_csv(self.metadata_file)
        
        # Проверяем наличие нужных колонок
        required_columns = ['image', 'gaze_x', 'gaze_y', 'gaze_h_original', 'gaze_v_original']
        for col in required_columns:
            if col not in metadata.columns:
                raise ValueError(f"Отсутствует обязательная колонка: {col}")
        
        print(f"Всего изображений: {len(metadata)}")
        
        # Разделяем по людям (чтобы один человек не был в разных наборах)
        if 'person_id' in metadata.columns:
            person_ids = metadata['person_id'].unique()
            train_persons, temp_persons = train_test_split(person_ids, test_size=0.3, random_state=42)
            val_persons, test_persons = train_test_split(temp_persons, test_size=0.5, random_state=42)
            
            train_mask = metadata['person_id'].isin(train_persons)
            val_mask = metadata['person_id'].isin(val_persons)
            test_mask = metadata['person_id'].isin(test_persons)
            
            train_df = metadata[train_mask].copy()
            val_df = metadata[val_mask].copy()
            test_df = metadata[test_mask].copy()
        else:
            # Если нет person_id, просто случайно разделяем
            train_df, temp_df = train_test_split(metadata, test_size=0.3, random_state=42)
            val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
        
        print(f"Train set: {len(train_df)} изображений")
        print(f"Validation set: {len(val_df)} изображений")
        print(f"Test set: {len(test_df)} изображений")
        
        # Сохраняем разделенные данные
        train_df.to_csv(os.path.join(self.data_dir, 'train_metadata.csv'), index=False)
        val_df.to_csv(os.path.join(self.data_dir, 'val_metadata.csv'), index=False)
        test_df.to_csv(os.path.join(self.data_dir, 'test_metadata.csv'), index=False)
        
        # Аугментации для тренировочного набора
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        
        # Трансформации для валидации и теста
        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        
        # Создание датасетов
        train_dataset = ColumbiaGazeDataset(
            train_df, self.data_dir, train_transform
        )
        
        val_dataset = ColumbiaGazeDataset(
            val_df, self.data_dir, val_transform
        )
        
        test_dataset = ColumbiaGazeDataset(
            test_df, self.data_dir, val_transform
        )
        
        return train_dataset, val_dataset, test_dataset
    
    def train_epoch(self, train_loader, epoch):
        """Обучение на одной эпохе"""
        self.model.train()
        running_loss = 0.0
        batch_count = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")
        for batch_idx, (images, gazes, _, _) in enumerate(progress_bar):
            images = images.to(self.device)
            gazes = gazes.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, gazes)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            running_loss += loss.item()
            batch_count += 1
            
            # Обновляем progress bar
            if batch_idx % 10 == 0:
                progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = running_loss / batch_count
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    def validate(self, val_loader, epoch):
        """Валидация модели"""
        self.model.eval()
        running_loss = 0.0
        batch_count = 0
        
        # Для расчета ошибки в градусах
        all_errors = []
        
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]")
            for batch_idx, (images, gazes, gaze_h_orig, gaze_v_orig) in enumerate(progress_bar):
                images = images.to(self.device)
                gazes = gazes.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, gazes)
                
                running_loss += loss.item()
                batch_count += 1
                
                # Вычисляем ошибку в градусах
                outputs_np = outputs.cpu().numpy()
                pred_h_deg = outputs_np[:, 0] * 15  # денормализация горизонтального
                pred_v_deg = outputs_np[:, 1] * 20  # денормализация вертикального
                
                true_h_deg = gaze_h_orig.numpy().flatten()
                true_v_deg = gaze_v_orig.numpy().flatten()
                
                # Евклидова ошибка в градусах
                errors = np.sqrt((pred_h_deg - true_h_deg)**2 + (pred_v_deg - true_v_deg)**2)
                all_errors.extend(errors)
                
                if batch_idx % 10 == 0:
                    progress_bar.set_postfix({'loss': loss.item(), 'error_deg': np.mean(errors)})
        
        avg_loss = running_loss / batch_count
        avg_error_deg = np.mean(all_errors) if all_errors else 0
        
        self.val_losses.append(avg_loss)
        self.val_errors.append(avg_error_deg)
        
        return avg_loss, avg_error_deg
    
    def train(self, num_epochs=20, batch_size=32):
        """Основной цикл обучения"""
        print("Начало обучения модели...")
        
        # Подготовка данных
        train_dataset, val_dataset, test_dataset = self.prepare_datasets()
        
        # Создание DataLoader'ов
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=0, pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=0, pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=0, pin_memory=True
        )
        
        print(f"Размер батча: {batch_size}")
        print(f"Количество эпох: {num_epochs}")
        
        best_val_error = float('inf')
        patience_counter = 0
        patience = 5  # Ранняя остановка
        
        # Цикл обучения
        for epoch in range(num_epochs):
            print(f"\n{'='*60}")
            print(f"Эпоха {epoch+1}/{num_epochs}")
            print(f"{'='*60}")
            
            # Обучение
            train_loss = self.train_epoch(train_loader, epoch)
            
            # Валидация
            val_loss, val_error = self.validate(val_loader, epoch)
            
            # Обновление learning rate
            self.scheduler.step()
            
            # Вывод результатов эпохи
            print(f"\nРезультаты эпохи {epoch+1}:")
            print(f"  Train Loss:     {train_loss:.6f}")
            print(f"  Val Loss:       {val_loss:.6f}")
            print(f"  Val Error:      {val_error:.2f}°")
            print(f"  Learning Rate:  {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Сохранение лучшей модели
            if val_error < best_val_error:
                best_val_error = val_error
                patience_counter = 0
                
                # Сохраняем модель в простом формате (только веса)
                self.save_model('columbia_gaze_model.pt', epoch, val_error)
                print(f"  ✓ Модель сохранена! (ошибка: {val_error:.2f}°)")
            else:
                patience_counter += 1
                print(f"  Паттерн: {patience_counter}/{patience}")
                
                # Ранняя остановка
                if patience_counter >= patience:
                    print(f"\nРанняя остановка на эпохе {epoch+1}")
                    break
        
        # Тестирование на тестовом наборе
        print(f"\n{'='*60}")
        print("Тестирование на тестовом наборе...")
        test_loss, test_error = self.validate(test_loader, epoch)
        print(f"Test Loss:  {test_loss:.6f}")
        print(f"Test Error: {test_error:.2f}°")
        
        # Визуализация результатов
        self.plot_training_history()
        
        print("\nОбучение завершено!")
        print(f"Лучшая ошибка на валидации: {best_val_error:.2f}°")
        
        return best_val_error
    
    # В классе ColumbiaGazeTrainer исправьте метод save_model:

    def save_model(self, filename, epoch, error):
        """Сохранение модели в правильном формате"""
        # Сохраняем ТОЛЬКО веса модели (без дополнительных метаданных с numpy)
        # Используем простой формат с state_dict
        torch.save(self.model.state_dict(), filename)
    
        # Отдельно сохраняем метаданные в текстовом формате
        with open(filename.replace('.pt', '_info.txt'), 'w') as f:
            f.write(f"Epoch: {epoch}\n")
            f.write(f"Validation Error: {error:.2f}°\n")
            f.write(f"Model: EfficientGazeNet\n")
            f.write(f"Input size: 224x224\n")
            f.write(f"Output range: tanh [-1, 1]\n")
            f.write(f"Normalization: gaze_h/15, gaze_v/20\n")
    
    def plot_training_history(self):
        """Визуализация истории обучения"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # График потерь
        axes[0].plot(self.train_losses, label='Train Loss', linewidth=2, marker='o')
        axes[0].plot(self.val_losses, label='Validation Loss', linewidth=2, marker='s')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss (MSE)')
        axes[0].set_title('Loss during training')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # График ошибки в градусах
        axes[1].plot(self.val_errors, label='Validation Error', linewidth=2, marker='^', color='red')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Error (degrees)')
        axes[1].set_title('Gaze estimation error')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('columbia_training_history.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # Сохраняем историю в CSV
        history_df = pd.DataFrame({
            'epoch': range(1, len(self.train_losses) + 1),
            'train_loss': self.train_losses,
            'val_loss': self.val_losses,
            'val_error_deg': self.val_errors
        })
        history_df.to_csv('training_history.csv', index=False)
        print("История обучения сохранена в training_history.csv")

def create_test_dataset_if_needed():
    """Создание тестового датасета если реальные данные не загружены"""
    print("Создание тестового датасета...")
    
    os.makedirs('test_dataset', exist_ok=True)
    
    test_data = []
    
    # Создаем 200 тестовых изображений с разными направлениями взгляда
    for i in range(200):
        # Создаем простое изображение лица
        img = np.zeros((224, 224, 3), dtype=np.uint8)
        
        # Рисуем лицо
        cv2.rectangle(img, (50, 50), (174, 174), (200, 200, 200), -1)  # Лицо
        cv2.circle(img, (90, 90), 15, (255, 255, 255), -1)  # Левый глаз
        cv2.circle(img, (134, 90), 15, (255, 255, 255), -1)  # Правый глаз
        
        # Направление взгляда (равномерное распределение)
        gaze_h = (i % 15) - 7  # -7 до +7
        gaze_v = (i // 15) - 6  # -6 до +6
        
        # Нормализация
        gaze_h_norm = gaze_h / 15.0
        gaze_v_norm = gaze_v / 20.0
        
        # Рисуем зрачки в зависимости от направления взгляда
        pupil_offset_h = int(gaze_h_norm * 8)
        pupil_offset_v = int(gaze_v_norm * 6)
        
        cv2.circle(img, (90 + pupil_offset_h, 90 + pupil_offset_v), 5, (0, 0, 0), -1)
        cv2.circle(img, (134 + pupil_offset_h, 90 + pupil_offset_v), 5, (0, 0, 0), -1)
        
        # Рисуем рот
        cv2.ellipse(img, (112, 140), (30, 15), 0, 0, 180, (0, 0, 0), 2)
        
        # Сохраняем изображение
        img_filename = f'test_{i:04d}.jpg'
        img_path = os.path.join('test_dataset', img_filename)
        cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        
        test_data.append({
            'image': img_filename,
            'gaze_x': float(gaze_h_norm),
            'gaze_y': float(gaze_v_norm),
            'gaze_h_original': float(gaze_h),
            'gaze_v_original': float(gaze_v),
            'person_id': f'person_{i % 10:02d}'
        })
    
    # Сохраняем метаданные
    df = pd.DataFrame(test_data)
    df.to_csv('test_dataset/metadata.csv', index=False)
    
    print(f"Создано {len(test_data)} тестовых изображений")
    return 'test_dataset', 'test_dataset/metadata.csv'

def main():
    """Основная функция"""
    print("Columbia Gaze Model Training")
    print("=" * 50)
    
    # Путь к данным
    data_dir = 'columbia_processed'
    metadata_file = os.path.join(data_dir, 'metadata.csv')
    
    # Проверяем наличие реальных данных
    if not os.path.exists(metadata_file):
        print(f"Реальные данные не найдены в {metadata_file}")
        print("Создание тестового датасета...")
        data_dir, metadata_file = create_test_dataset_if_needed()
    else:
        print(f"Найдены реальные данные: {metadata_file}")
    
    try:
        # Создаем тренер и запускаем обучение
        trainer = ColumbiaGazeTrainer(data_dir, metadata_file)
        best_error = trainer.train(num_epochs=15, batch_size=16)
        
        print(f"\nЛучшая модель сохранена как 'columbia_gaze_model.pt'")
        print(f"Ошибка предсказания: {best_error:.2f}°")
        
        # Инструкция для использования
        print("\nИнструкция по использованию:")
        print("1. Загрузите модель в приложении:")
        print("   model = EfficientGazeNet()")
        print("   checkpoint = torch.load('columbia_gaze_model.pt', map_location='cpu')")
        print("   model.load_state_dict(checkpoint['model_state_dict'])")
        print("2. Используйте для предсказания:")
        print("   gaze_normalized = model(image_tensor)")
        print("   gaze_degrees = [gaze_normalized[0]*15, gaze_normalized[1]*20]")
        
    except Exception as e:
        print(f"\nОшибка при обучении: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()