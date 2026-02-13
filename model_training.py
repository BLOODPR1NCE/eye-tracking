# model_training_simple.py
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os

def create_simple_model():
    """Создает простую модель для определения взгляда"""
    
    # Входы для левого и правого глаза
    left_input = keras.Input(shape=(36, 36, 1), name='left_eye')
    right_input = keras.Input(shape=(36, 36, 1), name='right_eye')
    
    # Общая ветвь для глаз
    def eye_branch():
        return keras.Sequential([
            keras.layers.Conv2D(32, 3, activation='relu'),
            keras.layers.MaxPooling2D(2),
            keras.layers.Conv2D(64, 3, activation='relu'),
            keras.layers.MaxPooling2D(2),
            keras.layers.Flatten(),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(0.3)
        ])
    
    # Обработка глаз
    left_features = eye_branch()(left_input)
    right_features = eye_branch()(right_input)
    
    # Объединение
    combined = keras.layers.Concatenate()([left_features, right_features])
    
    # Полносвязные слои
    x = keras.layers.Dense(256, activation='relu')(combined)
    x = keras.layers.Dropout(0.4)(x)
    x = keras.layers.Dense(128, activation='relu')(x)
    x = keras.layers.Dropout(0.3)(x)
    
    # Выход (X, Y координаты)
    output = keras.layers.Dense(2, activation='sigmoid', name='gaze')(x)
    
    return keras.Model(inputs=[left_input, right_input], outputs=output)

def main():
    print("Обучение модели для определения взгляда")
    
    # Загрузка данных
    data_dir = 'simple_data'
    left_train = np.load(f'{data_dir}/left_train.npy').reshape(-1, 36, 36, 1)
    right_train = np.load(f'{data_dir}/right_train.npy').reshape(-1, 36, 36, 1)
    y_train = np.load(f'{data_dir}/y_train.npy')
    
    left_val = np.load(f'{data_dir}/left_val.npy').reshape(-1, 36, 36, 1)
    right_val = np.load(f'{data_dir}/right_val.npy').reshape(-1, 36, 36, 1)
    y_val = np.load(f'{data_dir}/y_val.npy')
    
    left_test = np.load(f'{data_dir}/left_test.npy').reshape(-1, 36, 36, 1)
    right_test = np.load(f'{data_dir}/right_test.npy').reshape(-1, 36, 36, 1)
    y_test = np.load(f'{data_dir}/y_test.npy')
    
    # Создание модели
    model = create_simple_model()
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae', 'mse']
    )
    
    # Callback'и
    callbacks = [
        keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint('simple_model.keras', save_best_only=True),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
    ]
    
    # Обучение
    history = model.fit(
        [left_train, right_train], y_train,
        validation_data=([left_val, right_val], y_val),
        epochs=50,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    # Оценка
    test_loss = model.evaluate([left_test, right_test], y_test, verbose=0)
    print(f"\nTest Loss: {test_loss[0]:.4f}, Test MAE: {test_loss[1]:.4f}")
    
    # Сохранение
    model.save('gaze_model_simple.keras')
    print("Модель сохранена как 'gaze_model_simple.keras'")

if __name__ == "__main__":
    main()