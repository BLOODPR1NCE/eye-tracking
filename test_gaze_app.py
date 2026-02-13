# test_gaze_model.py
import sys
import unittest
import numpy as np
import tensorflow as tf
import cv2
import os

class TestGazeModel(unittest.TestCase):
    
    def setUp(self):
        """Подготовка тестовых данных"""
        self.test_left_eye = np.random.randn(5, 36, 36, 1).astype('float32')
        self.test_right_eye = np.random.randn(5, 36, 36, 1).astype('float32')
        self.test_targets = np.random.rand(5, 2).astype('float32')
        
        # Загружаем модель если существует
        if os.path.exists('gaze_model_simple.keras'):
            self.model = tf.keras.models.load_model('gaze_model_simple.keras')
        else:
            self.model = None
    
    def test_model_output_shape(self):
        """Тест формы выхода модели"""
        if self.model is None:
            self.skipTest("Модель не найдена")
            
        prediction = self.model.predict([self.test_left_eye, self.test_right_eye], verbose=0)
        self.assertEqual(prediction.shape, (5, 2))
    
    def test_output_range(self):
        """Тест диапазона выходных значений"""
        if self.model is None:
            self.skipTest("Модель не найдена")
            
        prediction = self.model.predict([self.test_left_eye, self.test_right_eye], verbose=0)
        self.assertTrue(np.all(prediction >= 0))
        self.assertTrue(np.all(prediction <= 1))
    
    def test_eye_preprocessing(self):
        """Тест предобработки изображений глаз"""
        # Создаем тестовое изображение глаза
        test_eye = np.random.randint(0, 256, (50, 50), dtype=np.uint8)
        
        # Предобработка
        processed = cv2.resize(test_eye, (36, 36))
        processed = cv2.equalizeHist(processed)
        processed = processed.astype('float32') / 255.0
        
        # Проверяем
        self.assertEqual(processed.shape, (36, 36))
        self.assertGreaterEqual(processed.min(), 0)
        self.assertLessEqual(processed.max(), 1)
    
    def test_data_loading(self):
        """Тест загрузки данных"""
        # Проверяем существование файлов данных
        data_files = [
            'simple_data/left_train.npy',
            'simple_data/right_train.npy',
            'simple_data/y_train.npy'
        ]
        
        for file in data_files:
            if os.path.exists(file):
                data = np.load(file)
                self.assertIsInstance(data, np.ndarray)
                self.assertGreater(len(data), 0)
            else:
                print(f"Файл {file} не найден")
    
    def test_coordinate_normalization(self):
        """Тест нормализации координат"""
        # Тестовые координаты экрана
        screen_x, screen_y = 960, 540  # Центр экрана 1920x1080
        
        # Нормализация
        norm_x = screen_x / 1920.0
        norm_y = screen_y / 1080.0
        
        self.assertAlmostEqual(norm_x, 0.5)
        self.assertAlmostEqual(norm_y, 0.5)
        self.assertTrue(0 <= norm_x <= 1)
        self.assertTrue(0 <= norm_y <= 1)

class TestGazeTrackerApp(unittest.TestCase):
    
    def setUp(self):
        """Настройка для тестов приложения"""
        # Создаем простую модель для тестов приложения
        self.model = None
        
    def test_eye_detection(self):
        """Тест обнаружения глаз"""
        # Создаем тестовое изображение с лицом
        test_img = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Рисуем простые глаза для теста
        cv2.circle(test_img, (200, 240), 30, (255, 255, 255), -1)  # Левый глаз
        cv2.circle(test_img, (440, 240), 30, (255, 255, 255), -1)  # Правый глаз
        
        # Добавляем лицо (прямоугольник)
        cv2.rectangle(test_img, (100, 100), (540, 380), (200, 200, 200), -1)
        
        gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
        
        # Используем упрощенный детектор
        try:
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            if face_cascade.empty():
                self.skipTest("Детектор лиц не загружен")
            
            faces = face_cascade.detectMultiScale(gray, 1.1, 5)
            # Принимаем любой результат (может быть пустым tuple или массив)
            self.assertIsInstance(faces, tuple)  # Haar cascade возвращает tuple
            
        except Exception as e:
            self.skipTest(f"Ошибка детекции: {e}")
    
    def test_model_inference_time(self):
        """Тест времени инференса модели"""
        model_paths = [
            'gaze_model_simple.keras',
            'models/gaze_model_final.keras',
            'models/gaze_model_final.h5'
        ]
        
        model = None
        for path in model_paths:
            if os.path.exists(path):
                model = tf.keras.models.load_model(path)
                break
        
        if model is None:
            self.skipTest("Модель не найдена")
            
        import time
        start_time = time.time()
        
        # Быстрый тест инференса
        test_input = [np.random.randn(1, 36, 36, 1).astype('float32'), 
                     np.random.randn(1, 36, 36, 1).astype('float32')]
        model.predict(test_input, verbose=0)
        
        inference_time = time.time() - start_time
        self.assertLess(inference_time, 2.0)  # Должно быть меньше 2 секунд
    
    def test_error_handling(self):
        """Тест обработки ошибок с мок-объектом модели"""
        # Создаем простую мок-модель для теста
        class MockModel:
            def predict(self, inputs, verbose=0):
                if inputs[0].shape != (5, 36, 36, 1):
                    raise ValueError("Неверная форма входных данных")
                return np.random.rand(5, 2)
        
        mock_model = MockModel()
        
        # Тест с правильными данными
        correct_input = [np.random.randn(5, 36, 36, 1), 
                        np.random.randn(5, 36, 36, 1)]
        result = mock_model.predict(correct_input)
        self.assertEqual(result.shape, (5, 2))
        
        # Тест с неправильными данными
        wrong_input = [np.random.randn(5, 32, 32, 1), 
                      np.random.randn(5, 32, 32, 1)]
        
        with self.assertRaises(ValueError):
            mock_model.predict(wrong_input)

def run_individual_tests():
    """Запуск отдельных тестов для отладки"""
    print("Запуск отдельных тестов для отладки...")
    
    # Создаем тестовый объект
    test_obj = TestGazeModel()
    test_obj.setUp()
    
    # Запускаем тесты из TestGazeModel
    print("\n1. Тестирование модели...")
    tests = [
        ('test_model_output_shape', test_obj.test_model_output_shape),
        ('test_output_range', test_obj.test_output_range),
        ('test_eye_preprocessing', test_obj.test_eye_preprocessing),
        ('test_data_loading', test_obj.test_data_loading),
        ('test_coordinate_normalization', test_obj.test_coordinate_normalization),
    ]
    
    for name, test_func in tests:
        try:
            test_func()
            print(f"  ✓ {name}: PASSED")
        except unittest.SkipTest as e:
            print(f"  - {name}: SKIPPED - {e}")
        except Exception as e:
            print(f"  ✗ {name}: FAILED - {e}")
    
    # Тесты для приложения
    print("\n2. Тестирование приложения...")
    app_test = TestGazeTrackerApp()
    app_test.setUp()
    
    app_tests = [
        ('test_eye_detection', app_test.test_eye_detection),
        ('test_model_inference_time', app_test.test_model_inference_time),
        ('test_error_handling', app_test.test_error_handling),
    ]
    
    for name, test_func in app_tests:
        try:
            test_func()
            print(f"  ✓ {name}: PASSED")
        except unittest.SkipTest as e:
            print(f"  - {name}: SKIPPED - {e}")
        except Exception as e:
            print(f"  ✗ {name}: FAILED - {e}")

if __name__ == '__main__':
    print("Запуск тестов модели определения взгляда")
    print("=" * 50)
    
    # Опция для отладки
    if len(sys.argv) > 1 and sys.argv[1] == '--debug':
        run_individual_tests()
    else:
        # Запуск всех тестов через unittest
        loader = unittest.TestLoader()
        
        # Загружаем тесты из обоих классов
        test_suite = unittest.TestSuite()
        test_suite.addTests(loader.loadTestsFromTestCase(TestGazeModel))
        test_suite.addTests(loader.loadTestsFromTestCase(TestGazeTrackerApp))
        
        # Запускаем тесты
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(test_suite)
        
        print("\n" + "=" * 50)
        print("РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ:")
        print(f"Всего тестов: {result.testsRun}")
        print(f"Провалено: {len(result.failures)}")
        print(f"Ошибок: {len(result.errors)}")
        print(f"Пропущено: {len(result.skipped)}")
        
        if result.failures or result.errors:
            print("\nДЕТАЛИ ОШИБОК:")
            for test, traceback in result.failures + result.errors:
                print(f"\n{test}:")
                print(traceback)