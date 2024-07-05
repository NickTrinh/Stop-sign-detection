import torch
from ultralytics import YOLO

def run():

    model = YOLO("yolov8n.pt")  # build a new model from scratch

    model.train(data="data/data.yaml", epochs=100, device=0, batch=-1)  # train the model

if __name__ == '__main__':
    run()
