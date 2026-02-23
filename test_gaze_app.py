# test_simple.py
import unittest
import numpy as np
import cv2
import os
import json
import requests
from PIL import Image
import io
import base64

class TestGazeSystem(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """Подготовка тестовых данных перед всеми тестами""" #фото, лицо, 2 глаза
        cls.test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(cls.test_image, (200, 150), (440, 330), (200, 200, 200), -1)
        cv2.circle(cls.test_image, (280, 240), 20, (0, 0, 0), -1)
        cv2.circle(cls.test_image, (360, 240), 20, (0, 0, 0), -1)
        
        cv2.imwrite('test_face.jpg', cls.test_image)
        
        # Создаем тестовое изображение без лица
        cls.test_image_no_face = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.imwrite('test_no_face.jpg', cls.test_image_no_face)
    
    @classmethod
    def tearDownClass(cls):
        """Очистка после всех тестов"""
        if os.path.exists('test_face.jpg'):
            os.remove('test_face.jpg')
        if os.path.exists('test_no_face.jpg'):
            os.remove('test_no_face.jpg')
    
    def test_1_model_exists(self):
        print("\n" + "="*60)
        print("ТЕСТ 1: Проверка наличия модели")
        print("="*60)
        
        model_path = 'models/gaze_model_final.keras'
        self.assertTrue(os.path.exists(model_path), f"Модель не найдена по пути: {model_path}")
        print(f"✓ Модель найдена: {model_path}")
    
    def test_2_data_exists(self):
        print("\n" + "="*60)
        print("ТЕСТ 2: Проверка наличия данных")
        print("="*60)
        
        required_files = [
            'simple_data/left_train.npy',
            'simple_data/right_train.npy',
            'simple_data/y_train.npy',
            'simple_data/left_val.npy',
            'simple_data/right_val.npy',
            'simple_data/y_val.npy',
            'simple_data/left_test.npy',
            'simple_data/right_test.npy',
            'simple_data/y_test.npy'
        ]
        
        for file_path in required_files:
            self.assertTrue(os.path.exists(file_path), 
                          f"Файл не найден: {file_path}")
            print(f"✓ Файл найден: {file_path}")
    
    def test_3_api_health(self):
        print("\n" + "="*60)
        print("ТЕСТ 3: Проверка API (health)")
        print("="*60)
        
        try:
            response = requests.get("http://localhost:5000/health", timeout=2)
            self.assertEqual(response.status_code, 200, f"Неверный статус код: {response.status_code}")
            
            data = response.json()
            self.assertIn('status', data, "Ответ не содержит поле 'status'")
            self.assertIn('model_loaded', data, "Ответ не содержит поле 'model_loaded'")
            
            print(f"✓ Статус API: {data['status']}")
            print(f"✓ Модель загружена: {data['model_loaded']}")
            
        except requests.exceptions.ConnectionError:
            self.skipTest("API сервер не запущен. Запустите gaze_api_simple.py")
        except Exception as e:
            self.fail(f"Ошибка при тестировании API: {e}")
    
    def test_4_model_prediction(self):
        print("\n" + "="*60)
        print("ТЕСТ 4: Предсказание модели")
        print("="*60)
        
        model_path = 'models/gaze_model_final.keras'
        if not os.path.exists(model_path):
            self.skipTest("Модель не найдена")
        
        import tensorflow as tf
        model = tf.keras.models.load_model(model_path)
        
        test_left = np.random.randn(1, 36, 36, 1).astype('float32')
        test_right = np.random.randn(1, 36, 36, 1).astype('float32')
        
        try:
            prediction = model.predict([test_left, test_right], verbose=0)
            
            self.assertEqual(prediction.shape, (1, 2), f"Неправильная форма выхода: {prediction.shape}")
            print(f"✓ Форма выхода: {prediction.shape}")
            
            self.assertTrue(np.all(prediction >= 0) and np.all(prediction <= 1), f"Выход за пределами [0,1]: {prediction}")
            print(f"✓ Диапазон предсказаний: [{prediction.min():.3f}, {prediction.max():.3f}]")
            
            self.assertFalse(np.isnan(prediction).any(), "Предсказание содержит NaN")
            print("✓ Предсказание не содержит NaN")
            
        except Exception as e:
            self.fail(f"Ошибка при предсказании: {e}")

def run_tests():
    print("\n" + "="*70)
    print("ЗАПУСК МОДУЛЬНЫХ ТЕСТОВ СИСТЕМЫ ОПРЕДЕЛЕНИЯ ВЗГЛЯДА")
    print("="*70)
    
    suite = unittest.TestLoader().loadTestsFromTestCase(TestGazeSystem)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "="*70)
    print("РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ")
    print("="*70)
    print(f"Всего тестов: {result.testsRun}")
    print(f"Успешно: {result.testsRun - len(result.failures) - len(result.errors) - len(result.skipped)}")
    print(f"Провалено: {len(result.failures)}")
    print(f"Ошибок: {len(result.errors)}")
    print(f"Пропущено: {len(result.skipped)}")
    
    if result.failures or result.errors:
        print("\nДЕТАЛИ ОШИБОК:")
        for test, traceback in result.failures + result.errors:
            print(f"\n{test}:")
            print(traceback)
    
    return result

if __name__ == '__main__':
    run_tests()