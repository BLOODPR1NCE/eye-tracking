# model_training.py
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import os
import json
from sklearn.model_selection import train_test_split

print("="*60)
print("ОБУЧЕНИЕ УЛУЧШЕННОЙ МОДЕЛИ ДЛЯ ОЦЕНКИ ВЗГЛЯДА")
print("="*60)

print("\nЗагрузка данных...")
data_dir = 'processed_data'

required_files = [
    'mpiigaze_full_left_train.npy',
    'mpiigaze_full_right_train.npy', 
    'mpiigaze_full_targets_train.npy',
    'mpiigaze_full_left_val.npy',
    'mpiigaze_full_right_val.npy',
    'mpiigaze_full_targets_val.npy',
    'mpiigaze_full_left_test.npy',
    'mpiigaze_full_right_test.npy',
    'mpiigaze_full_targets_test.npy'
]

for file in required_files:
    if not os.path.exists(os.path.join(data_dir, file)):
        print(f"Ошибка: файл {file} не найден!")
        exit(1)

left_train = np.load(os.path.join(data_dir, 'mpiigaze_full_left_train.npy'))
right_train = np.load(os.path.join(data_dir, 'mpiigaze_full_right_train.npy'))
y_train = np.load(os.path.join(data_dir, 'mpiigaze_full_targets_train.npy'))

left_val = np.load(os.path.join(data_dir, 'mpiigaze_full_left_val.npy'))
right_val = np.load(os.path.join(data_dir, 'mpiigaze_full_right_val.npy'))
y_val = np.load(os.path.join(data_dir, 'mpiigaze_full_targets_val.npy'))

left_test = np.load(os.path.join(data_dir, 'mpiigaze_full_left_test.npy'))
right_test = np.load(os.path.join(data_dir, 'mpiigaze_full_right_test.npy'))
y_test = np.load(os.path.join(data_dir, 'mpiigaze_full_targets_test.npy'))

with open(os.path.join(data_dir, 'mpiigaze_full_stats.json'), 'r') as f:
    stats = json.load(f)

print(f"\nДанные загружены:")
print(f"  Train: {len(left_train)} samples")
print(f"  Validation: {len(left_val)} samples")
print(f"  Test: {len(left_test)} samples")
print(f"  Форма изображений: {left_train.shape[1:]}")
print(f"  Диапазон целей: {stats['targets_min']} - {stats['targets_max']}")

print("\nСоздание улучшенной модели...")

def create_improved_gaze_model(input_shape=(36, 36, 1)):
    left_input = keras.layers.Input(shape=input_shape, name='left_eye_input')
    right_input = keras.layers.Input(shape=input_shape, name='right_eye_input')
    
    def create_eye_branch():
        model = keras.Sequential([
            keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D((2, 2)),
            
            keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D((2, 2)),
            
            keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D((2, 2)),
            
            keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.GlobalAveragePooling2D(),
            
            keras.layers.Dense(128, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.3),
            
            keras.layers.Dense(64, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.2)
        ])
        return model
    
    eye_branch = create_eye_branch() #ветви для глаз
    
    left_features = eye_branch(left_input) #обработка обоих глаз
    right_features = eye_branch(right_input)
    
    merged = keras.layers.Concatenate()([left_features, right_features])
    
    x = keras.layers.Dense(256, activation='relu')(merged)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.4)(x)
    
    x = keras.layers.Dense(128, activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.3)(x)
    
    x = keras.layers.Dense(64, activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.2)(x)
    
    output = keras.layers.Dense(2, activation='sigmoid', name='gaze_output')(x)
    
    model = keras.Model(inputs=[left_input, right_input], outputs=output)
    return model

model = create_improved_gaze_model()

initial_learning_rate = 0.001
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=1000,
    decay_rate=0.96,
    staircase=True
)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
    loss='mse',
    metrics=[
        'mae',
        keras.metrics.RootMeanSquaredError(name='rmse')
    ]
)

print("\nАрхитектура модели:")
model.summary()

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        verbose=1,
        min_delta=0.0001
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=1e-7,
        verbose=1
    ),
    keras.callbacks.ModelCheckpoint(
        os.path.join('models', 'best_gaze_model.keras'),
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    ),
    keras.callbacks.TensorBoard(
        log_dir='logs',
        histogram_freq=1,
        write_graph=True
    ),
    keras.callbacks.CSVLogger('training_log.csv')
]

os.makedirs('models', exist_ok=True)

print("\n" + "="*50)
print("НАЧАЛО ОБУЧЕНИЯ")
print("="*50)

batch_size = min(256, len(left_train) // 10)

history = model.fit(
    [left_train, right_train], y_train,
    validation_data=([left_val, right_val], y_val),
    epochs=30,
    batch_size=batch_size,
    callbacks=callbacks,
    verbose=1,
    shuffle=True
)

print("\n" + "="*50)
print("ОБУЧЕНИЕ ЗАВЕРШЕНО")
print("="*50)

print("\nОценка на тестовых данных...")
test_results = model.evaluate([left_test, right_test], y_test, verbose=0)

print(f"\nРезультаты на тестовых данных:")
print(f"  Loss (MSE): {test_results[0]:.6f}")
print(f"  MAE: {test_results[1]:.6f}")
print(f"  RMSE: {test_results[2]:.6f}")

# Предсказания на тестовых данных
print("\nПримеры предсказаний:")
predictions = model.predict([left_test[:5], right_test[:5]], verbose=0)

for i in range(5):
    pred = predictions[i]
    true = y_test[i]
    error_mse = np.mean((pred - true)**2)
    error_mae = np.mean(np.abs(pred - true))
    
    print(f"\nПример {i+1}:")
    print(f"  Предсказано: X={pred[0]:.4f}, Y={pred[1]:.4f}")
    print(f"  Истинное:    X={true[0]:.4f}, Y={true[1]:.4f}")
    print(f"  MSE: {error_mse:.6f}, MAE: {error_mae:.6f}")

fig, axes = plt.subplots(2, 3, figsize=(18, 10)) #график

axes[0, 0].plot(history.history['loss'], label='Train Loss', linewidth=2)
axes[0, 0].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
axes[0, 0].set_xlabel('Epoch', fontsize=12)
axes[0, 0].set_ylabel('Loss (MSE)', fontsize=12)
axes[0, 0].set_title('Функция потерь', fontsize=14)
axes[0, 0].legend(fontsize=10)
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_yscale('log')

axes[0, 1].plot(history.history['mae'], label='Train MAE', linewidth=2)
axes[0, 1].plot(history.history['val_mae'], label='Val MAE', linewidth=2)
axes[0, 1].set_xlabel('Epoch', fontsize=12)
axes[0, 1].set_ylabel('MAE', fontsize=12)
axes[0, 1].set_title('Средняя абсолютная ошибка', fontsize=14)
axes[0, 1].legend(fontsize=10)
axes[0, 1].grid(True, alpha=0.3)

axes[0, 2].plot(history.history['rmse'], label='Train RMSE', linewidth=2)
axes[0, 2].plot(history.history['val_rmse'], label='Val RMSE', linewidth=2)
axes[0, 2].set_xlabel('Epoch', fontsize=12)
axes[0, 2].set_ylabel('RMSE', fontsize=12)
axes[0, 2].set_title('Среднеквадратичная ошибка', fontsize=14)
axes[0, 2].legend(fontsize=10)
axes[0, 2].grid(True, alpha=0.3)

if 'lr' in history.history:
    axes[1, 0].plot(history.history['lr'], linewidth=2, color='purple')
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('Learning Rate', fontsize=12)
    axes[1, 0].set_title('Скорость обучения', fontsize=14)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_yscale('log')

# Распределение ошибок на тестовых данных
all_predictions = model.predict([left_test, right_test], batch_size=256, verbose=0)
errors = np.sqrt(np.sum((all_predictions - y_test)**2, axis=1))

axes[1, 1].hist(errors, bins=50, edgecolor='black', alpha=0.7, color='red')
axes[1, 1].set_xlabel('Ошибка (Euclidean distance)', fontsize=12)
axes[1, 1].set_ylabel('Частота', fontsize=12)
axes[1, 1].set_title('Распределение ошибок на тестовых данных', fontsize=14)
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].axvline(x=errors.mean(), color='blue', linestyle='--', 
                  label=f'Среднее: {errors.mean():.4f}')
axes[1, 1].legend()

# Сравнение предсказаний и истинных значений
axes[1, 2].scatter(y_test[:, 0], y_test[:, 1], alpha=0.3, s=10, label='Истинные', color='blue')
axes[1, 2].scatter(all_predictions[:, 0], all_predictions[:, 1], alpha=0.3, s=10, 
                  label='Предсказанные', color='red')
axes[1, 2].set_xlabel('X координата', fontsize=12)
axes[1, 2].set_ylabel('Y координата', fontsize=12)
axes[1, 2].set_title('Сравнение истинных и предсказанных значений', fontsize=14)
axes[1, 2].legend(fontsize=10)
axes[1, 2].grid(True, alpha=0.3)

plt.suptitle('Результаты обучения улучшенной модели', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig('models/training_results.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nСохранение моделей...")
model.save('models/gaze_model_final.keras')
model.save('models/gaze_model_final.h5')

np.save('models/training_history.npy', history.history)

print("\nМодели сохранены:")
print("  models/gaze_model_final.keras")
print("  models/gaze_model_final.h5")
print("  models/training_history.npy")
print("  models/training_results.png")

# Дополнительная статистика
print("\n" + "="*50)
print("СТАТИСТИКА МОДЕЛИ")
print("="*50)

# Расчет дополнительных метрик
test_errors = np.sqrt(np.sum((all_predictions - y_test)**2, axis=1))

print(f"\nСтатистика ошибок на тестовых данных:")
print(f"  Средняя ошибка: {test_errors.mean():.6f}")
print(f"  Медианная ошибка: {np.median(test_errors):.6f}")
print(f"  Стандартное отклонение: {test_errors.std():.6f}")
print(f"  Минимальная ошибка: {test_errors.min():.6f}")
print(f"  Максимальная ошибка: {test_errors.max():.6f}")

# Процентили ошибок
percentiles = [25, 50, 75, 90, 95, 99]
print(f"\nПроцентили ошибок:")
for p in percentiles:
    print(f"  {p}% ошибок меньше: {np.percentile(test_errors, p):.6f}")

print("\n" + "="*50)
print("МОДЕЛЬ ГОТОВА К ИСПОЛЬЗОВАНИЮ!")
print("="*50)