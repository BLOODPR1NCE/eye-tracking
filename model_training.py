#model_training.py
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

print("="*50)
print("ЗАГРУЗКА ДАННЫХ И ОБУЧЕНИЕ МОДЕЛИ")
print("="*50)

# Загружаем данные
print("\nЗагрузка данных...")
left_X = np.load('left_eyes_simple.npy')
right_X = np.load('right_eyes_simple.npy')
y = np.load('targets_simple.npy')

print(f"Данные загружены:")
print(f"  Левые глаза: {left_X.shape}")
print(f"  Правые глаза: {right_X.shape}")
print(f"  Цели: {y.shape}")
print(f"  Диапазон X: {y[:, 0].min():.3f} - {y[:, 0].max():.3f}")
print(f"  Диапазон Y: {y[:, 1].min():.3f} - {y[:, 1].max():.3f}")

# Разделяем на train/validation/test
indices = np.arange(len(left_X))
train_idx, temp_idx = train_test_split(indices, test_size=0.3, random_state=42)
val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)

left_train, left_val, left_test = left_X[train_idx], left_X[val_idx], left_X[test_idx]
right_train, right_val, right_test = right_X[train_idx], right_X[val_idx], right_X[test_idx]
y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

print(f"\nРазделение данных:")
print(f"  Обучающая выборка: {len(train_idx)}")
print(f"  Валидационная выборка: {len(val_idx)}")
print(f"  Тестовая выборка: {len(test_idx)}")

# Создаем модель
print("\nСоздание модели...")

def create_simple_eye_model():
    """Простая модель для обработки глаз"""
    left_input = keras.layers.Input(shape=(32, 32, 1), name='left_eye')
    right_input = keras.layers.Input(shape=(32, 32, 1), name='right_eye')
    
    # Общая архитектура для глаз
    def eye_network(x):
        x = keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        x = keras.layers.MaxPooling2D((2, 2))(x)
        
        x = keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
        x = keras.layers.MaxPooling2D((2, 2))(x)
        
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(32, activation='relu')(x)
        x = keras.layers.Dropout(0.2)(x)
        
        return x
    
    # Обрабатываем оба глаза
    left_features = eye_network(left_input)
    right_features = eye_network(right_input)
    
    # Объединяем
    merged = keras.layers.Concatenate()([left_features, right_features])
    
    # Дополнительные слои
    x = keras.layers.Dense(64, activation='relu')(merged)
    x = keras.layers.Dropout(0.3)(x)
    
    x = keras.layers.Dense(32, activation='relu')(x)
    x = keras.layers.Dropout(0.2)(x)
    
    # Выходной слой
    output = keras.layers.Dense(2, activation='sigmoid')(x)
    
    # Создаем модель
    model = keras.Model(inputs=[left_input, right_input], outputs=output)
    return model

# Создаем и компилируем модель
model = create_simple_eye_model()

# Компилируем БЕЗ метрики mse в виде строки
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss=keras.losses.MeanSquaredError(),  # Используем класс вместо строки
    metrics=[keras.metrics.MeanAbsoluteError()]  # MAE вместо 'mae'
)

print("\nАрхитектура модели:")
model.summary()

# Callbacks
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    ),
    keras.callbacks.ModelCheckpoint(
        'best_model.keras',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
]

# Обучение
print("\n" + "="*50)
print("НАЧАЛО ОБУЧЕНИЯ")
print("="*50)

history = model.fit(
    [left_train, right_train], y_train,
    validation_data=([left_val, right_val], y_val),
    epochs=50,
    batch_size=100,
    callbacks=callbacks,
    verbose=1
)

print("\n" + "="*50)
print("ОБУЧЕНИЕ ЗАВЕРШЕНО")
print("="*50)

# Оценка на тестовых данных
print("\nОценка на тестовых данных...")
test_results = model.evaluate([left_test, right_test], y_test, verbose=0)
print(f"Test Loss: {test_results[0]:.4f}")
print(f"Test MAE: {test_results[1]:.4f}")

# Предсказания
print("\nПримеры предсказаний:")
predictions = model.predict([left_test[:3], right_test[:3]], verbose=0)

for i in range(3):
    pred = predictions[i]
    true = y_test[i]
    error = np.sqrt(np.sum((pred - true)**2))
    
    print(f"\nПример {i+1}:")
    print(f"  Предсказано: [{pred[0]:.3f}, {pred[1]:.3f}]")
    print(f"  Истинное:    [{true[0]:.3f}, {true[1]:.3f}]")
    print(f"  Ошибка: {error:.3f}")

# Графики обучения
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(history.history['loss'], label='Train Loss', linewidth=2)
axes[0].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Loss', fontsize=12)
axes[0].set_title('Loss', fontsize=14)
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

axes[1].plot(history.history['mean_absolute_error'], label='Train MAE', linewidth=2)
axes[1].plot(history.history['val_mean_absolute_error'], label='Val MAE', linewidth=2)
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('MAE', fontsize=12)
axes[1].set_title('Mean Absolute Error', fontsize=14)
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_history_final.png', dpi=100, bbox_inches='tight')
plt.show()

# Сохранение модели
print("\nСохранение модели...")
model.save('gaze_model_final.keras')  # Используем .keras формат
print("Модель сохранена как 'gaze_model_final.keras'")

# Также сохраняем в формате .h5 для совместимости
model.save('gaze_model_final.h5')
print("Модель также сохранена как 'gaze_model_final.h5'")

print("\n" + "="*50)
print("МОДЕЛЬ ГОТОВА К ИСПОЛЬЗОВАНИЮ")
print("="*50)