import torch

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)  # Load a pretrained YOLOv5s model ('yolov5s' is the small version)

# Step 2: Define dataset path and parameters
imgs = ['M:/Datasets/WoodPile/train/images/00000_rdNwQ66oVJ_0CI0t2_600x450_jpg.rf.a483f0bf53b62367b3bf276c736422d5.jpg']  # batch of images

# Inference
results = model(imgs)

# Results
results.print()
results.show()