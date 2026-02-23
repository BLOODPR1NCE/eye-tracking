# data_preparation.py
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import os

class SimpleGazeDataLoader:
    def __init__(self, data_path):
        self.data_path = data_path
        
    def load_mpiigaze(self, max_participants=3):
        eyes_left, eyes_right, targets = [], [], []
        
        for p in range(max_participants):
            p_str = f'p{str(p).zfill(2)}' #p01
            path = os.path.join(self.data_path, 'Data', 'Original', p_str)
            
            if not os.path.exists(path):
                continue
                
            for day in os.listdir(path):
                day_path = os.path.join(path, day)
                ann_file = os.path.join(day_path, 'annotation.txt')
                
                if not os.path.exists(ann_file):
                    continue
                    
                try:
                    anns = np.loadtxt(ann_file)
                    for i, ann in enumerate(anns):
                        if len(ann) < 26:
                            continue
                            
                        img_path = self.find_image(day_path, i)
                        if not img_path:
                            continue
                            
                        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                        if img is None:
                            continue
                            
                        x, y = ann[24]/1920.0, ann[25]/1080.0
                        
                        left, right = self.extract_eyes(img)
                        if left is not None and right is not None:
                            left = self.prepare_eye(left)
                            right = self.prepare_eye(right)
                            
                            eyes_left.append(left)
                            eyes_right.append(right)
                            targets.append([x, y])
                except:
                    continue
                    
        return np.array(eyes_left), np.array(eyes_right), np.array(targets)
    
    def find_image(self, path, idx):
        exts = ['.jpg', '.jpeg', '.png']
        files = [f for f in os.listdir(path) 
                if any(f.lower().endswith(ext) for ext in exts)]
        files.sort()
        return os.path.join(path, files[idx]) if idx < len(files) else None
    
    def extract_eyes(self, img):
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            
        faces = face_cascade.detectMultiScale(img, 1.1, 5)
        
        if len(faces) == 0:
            return None, None
            
        x, y, w, h = faces[0]
        face_roi = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(face_roi, 1.1, 3)
        
        if len(eyes) < 2:
            return None, None
            
        eyes = sorted(eyes, key=lambda e: e[0])
        ex1, ey1, ew1, eh1 = eyes[0]
        ex2, ey2, ew2, eh2 = eyes[1]
        
        left = face_roi[ey1:ey1+eh1, ex1:ex1+ew1]
        right = face_roi[ey2:ey2+eh2, ex2:ex2+ew2]
        
        return left, right
    
    def prepare_eye(self, eye_img):
        eye = cv2.resize(eye_img, (36, 36))
        eye = cv2.equalizeHist(eye)
        return eye.astype('float32') / 255.0

def main():
    print("Подготовка данных MPIIGaze")
    loader = SimpleGazeDataLoader(r"C:\Users\prince\Downloads\MPIIGaze\MPIIGaze")
    left, right, y = loader.load_mpiigaze(max_participants=3)
    
    if len(left) == 0:
        print("Данные не загружены!")
        return
    
    left_train, left_test, right_train, right_test, y_train, y_test = train_test_split(
        left, right, y, test_size=0.2, random_state=42)
    
    left_train, left_val, right_train, right_val, y_train, y_val = train_test_split(
        left_train, right_train, y_train, test_size=0.2, random_state=42)
    
    os.makedirs('simple_data', exist_ok=True)
    np.save('simple_data/left_train.npy', left_train)
    np.save('simple_data/right_train.npy', right_train)
    np.save('simple_data/y_train.npy', y_train)
    
    np.save('simple_data/left_val.npy', left_val)
    np.save('simple_data/right_val.npy', right_val)
    np.save('simple_data/y_val.npy', y_val)
    
    np.save('simple_data/left_test.npy', left_test)
    np.save('simple_data/right_test.npy', right_test)
    np.save('simple_data/y_test.npy', y_test)
    
    print(f"Данные сохранены: {len(left_train)} train, {len(left_val)} val, {len(left_test)} test")

if __name__ == "__main__":
    main()