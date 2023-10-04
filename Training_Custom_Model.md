
# Training Models using Google Colab





Open a new Colab Notebook and follow the steps mentioned below
#
1: First step is to mount your Google Drive to Notebook 
```bash
from google.colab import drive
drive.mount('/content/drive')
```
#
2: Let's make sure that we have access to GPU. We can use nvidia-smi command to do that. In case of any problems navigate to Edit -> Notebook settings -> Hardware accelerator, set it to GPU, and then click Save.
```bash
!nvidia-smi
```
#
3: Install YOLOv8 using Ultralytics
```bash
!pip install ultralytics
from ultralytics import YOLO
```
#
4: To train the model on your custom dataset, run the following command replacing the text in "data=" with the path to your Dataset's data.yaml file
```bash
!yolo task=detect mode=train model=yolov8l.pt data=${dataset.location}/data.yaml epochs=50 imgsz=640
```
#### yolo mode=predict runs YOLOv8 inference on your dataset, downloading models automatically from the latest YOLOv8 release, and saving results to runs/predict.

5: From runs/predict download the best.pt file which is now ready to use for detection.
#
6: Run this command to save the model to a desired location in the Google Drive by replacing text in the torch.load() function with a location in your Google Drive.

```bash
import torch
model.load_state_dict(torch.load({LOCATION}/model.pth))
```


