# gaze_tracker_api.py
import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
import requests
import threading

class SimpleGazeTracker:
    def __init__(self):
        self.api_url = "http://localhost:5000"
        self.cap = cv2.VideoCapture(0)
        self.current_pred = (0.5, 0.5)
        
        self.root = tk.Tk()
        self.root.title("Gaze Tracker")
        
        self.canvas = tk.Canvas(self.root, width=1920, height=1080)
        self.canvas.pack()
        
        self.dot = self.canvas.create_oval(0, 0, 20, 20, fill='red')
        
        self.video_label = None
        
        self.update()
        self.root.mainloop()
    
    def call_api(self, frame):
        try:
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            files = {'image': ('frame.jpg', buffer.tobytes(), 'image/jpeg')}
            
            response = requests.post(f"{self.api_url}/predict", files=files, timeout=0.5)
            
            if response.status_code == 200:
                data = response.json()
                if 'gaze_point' in data:
                    self.current_pred = (data['gaze_point']['x'], data['gaze_point']['y'])
        except:
            pass
    
    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        threading.Thread(target=self.call_api, args=(frame.copy(),), daemon=True).start()
        
        x, y = int(self.current_pred[0] * 1920), int(self.current_pred[1] * 1080)
        self.canvas.coords(self.dot, x-10, y-10, x+10, y+10)
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        img_tk = ImageTk.PhotoImage(image=img)
        
        if self.video_label is None:
            self.video_label = tk.Label(self.root, image=img_tk)
            self.video_label.image = img_tk
            self.video_label.pack()
        else:
            self.video_label.config(image=img_tk)
            self.video_label.image = img_tk
        
        return frame
    
    def update(self):
        self.process_frame()
        self.root.after(30, self.update)
    
    def __del__(self):
        self.cap.release()

if __name__ == "__main__":
    try:
        requests.get("http://localhost:5000/health", timeout=1)
        print("API доступен, запуск приложения...")
    except:
        print("ВНИМАНИЕ: API не доступен! Запустите сначала gaze_api_simple.py")
    
    app = SimpleGazeTracker()