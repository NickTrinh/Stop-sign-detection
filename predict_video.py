import os
from ultralytics import YOLO
import cv2

VIDEOS_DIR = os.path.join('.', 'videos') # get test video path

video_path = os.path.join(VIDEOS_DIR, 'stop_sign_1.mp4')
video_path_out = '{}_out.mp4'.format(os.path.splitext(video_path)[0]) # output video

cap = cv2.VideoCapture(video_path) # in video
ret, frame = cap.read()
H, W, _ = frame.shape
out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'mp4v'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H)) # out video

model_path = os.path.join('.', 'runs', 'detect', 'train25', 'weights', 'best.pt')

model = YOLO(model_path)  # load the model

while ret:
    
    results = model.predict(frame)
    
    plotted_frame = results[0].plot()
    
    out.write(plotted_frame)
    ret, frame = cap.read()

cap.release()
out.release()
cv2.destroyAllWindows()
