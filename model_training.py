# gaze_model.py
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os

def create_simple_model():
    left_input = keras.Input(shape=(36, 36, 1), name='left_eye')
    right_input = keras.Input(shape=(36, 36, 1), name='right_eye')
    
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
    
    left_features = eye_branch()(left_input)
    right_features = eye_branch()(right_input)
    
    combined = keras.layers.Concatenate()([left_features, right_features])
    
    x = keras.layers.Dense(256, activation='relu')(combined)
    x = keras.layers.Dropout(0.4)(x)
    x = keras.layers.Dense(128, activation='relu')(x)
    
    output = keras.layers.Dense(2, activation='sigmoid', name='gaze')(x)
    
    return keras.Model(inputs=[left_input, right_input], outputs=output)

def train_model():
    print("Обучение модели...")
    
    data_dir = 'simple_data'
    if not os.path.exists(data_dir):
        print("Сначала запустите data_preparation_simple.py")
        return
    
    left_train = np.load(f'{data_dir}/left_train.npy').reshape(-1, 36, 36, 1)
    right_train = np.load(f'{data_dir}/right_train.npy').reshape(-1, 36, 36, 1)
    y_train = np.load(f'{data_dir}/y_train.npy')
    
    left_val = np.load(f'{data_dir}/left_val.npy').reshape(-1, 36, 36, 1)
    right_val = np.load(f'{data_dir}/right_val.npy').reshape(-1, 36, 36, 1)
    y_val = np.load(f'{data_dir}/y_val.npy')
    
    model = create_simple_model()
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    os.makedirs('models', exist_ok=True)
    history = model.fit(
        [left_train, right_train], y_train,
        validation_data=([left_val, right_val], y_val),
        epochs=30,
        batch_size=32,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=5),
            keras.callbacks.ModelCheckpoint('models/gaze_model.keras', save_best_only=True)
        ],
        verbose=1
    )
    
    model.save('models/gaze_model_final.keras')
    print("Модель сохранена")

if __name__ == "__main__":
    train_model()