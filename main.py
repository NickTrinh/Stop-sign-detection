import torch
from ultralytics import YOLO

def run():
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = YOLO("yolov8n.yaml")  # build a new model from scratch

    model.to(device) #Use GPU

    model.train(data="config.yaml", epochs=600)  # train the model

if __name__ == '__main__':
    run()
