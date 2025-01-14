from ultralytics import YOLO


if __name__ == '__main__':
    # Load a model
    model = YOLO("data/yolo11n.pt")  # load a pretrained model (recommended for training)

    # Use the model
    model.train(data="data/label.yaml", epochs=1000, batch=128)  # train the model