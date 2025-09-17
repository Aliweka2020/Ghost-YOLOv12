

<div align="center">
<h1>Ghost-YOLOv12</h1>
<h3>Bio-Inspired Underwater Robotic Vehicle for Marine Exploration and AI-Powered Fish Detection</h3>

[Ali Elhenidy](https://github.com/Aliweka2020)<sup>1</sup>, [Ahmed Sameh](https://people.ucas.ac.cn/~qxye?language=en)<sup>1</sup>

<sup>1</sup>  Mansoura University , Egypt 


 [Research Square](https://www.researchsquare.com/article/rs-6538108/v1)

</div>



<details>
  <summary>
   <font size="+10">Abstract</font>
  </summary>
Ghost-YOLV12 is proposed, which is an enhanced version of the  YOLOv12 deep learning model. Trained on the DeepFish dataset, the proposed model achieved a mean average precision (mAP50) of 97.8 and demonstrated robust performance under occlusion, turbidity, and low-light conditions. All evaluations were conducted in simulation environments, with hydrodynamic testing performed through CFD and fish detection validated through annotated datasets. While no physical prototype has been deployed yet, the design is fully scalable and structured for real-world fabrication. 
</details>

[![Watch the video](https://img.youtube.com/vi/VIDEO_ID/maxresdefault.jpg)](https://youtu.be/3Zs-LLSfgzw)


## Main Results

**Turbo (default)**:
| Model (det)                                                                              | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | Speed (ms) <br><sup>T4 TensorRT10<br> | params<br><sup>(M) | FLOPs<br><sup>(G) |
| :----------------------------------------------------------------------------------- | :-------------------: | :-------------------:| :------------------------------:| :-----------------:| :---------------:|
| [Ghost-CBAM-YOLO12m](https://drive.google.com/file/d/1WuFZkm-snOwEcApp1ZOIV0TWQU58Px8l/view?usp=sharing) | 640                   | 97.8.4                 | 1.60                            | 2.5                | 6.0               |
| [YOLO12m](https://drive.google.com/file/d/1z2kUELXWfPcoGO-2Vkl1tKnmOv5tYnU3/view?usp=sharing) | 640                   | 97.1                 | 2.42                            | 9.1                | 19.4              |
| [Ghost-YOLO12m](https://drive.google.com/file/d/1O1BdTvHSciFhN1y24O5qCV63e77m_c5U/view?usp=sharing) | 640                   | 95.5                 | 4.27                            | 19.6               | 59.8              |
| [Ghost(Head& Backbone)-YOLOv12](https://drive.google.com/file/d/1lYU8WrUbv8MF-HQAxibewWAdyU2OpS8O/view?usp=sharing) | 640                   | 91.8                 | 5.83                            | 26.5               | 82.4              |




## Installation
```
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu11torch2.2cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
conda create -n yolov12 python=3.11
conda activate yolov12
pip install -r requirements.txt
pip install -e .
```



## Training 
```python
from ultralytics import YOLO

model = YOLO('yolov12n.yaml')

# Train the model
results = model.train(
  data='fish.yaml',
  epochs=600, 
  batch=256, 
  imgsz=640,
  scale=0.5,  # S:0.9; M:0.9; L:0.9; X:0.9
  mosaic=1.0,
  mixup=0.0,  # S:0.05; M:0.15; L:0.15; X:0.2
  copy_paste=0.1,  # S:0.15; M:0.4; L:0.5; X:0.6
  device="0,1,2,3",
)

# Evaluate model performance on the validation set
metrics = model.val()

# Perform object detection on an image
results = model("path/to/image.jpg")
results[0].show()

```

## Acknowledgement

The code is based on [ultralytics](https://github.com/ultralytics/ultralytics). Thanks for their excellent work!

## Citation

```
Ahmed Sameh, Ali Elhenidy. Bio-Inspired Underwater Robotic Vehicle for Marine Exploration and AI-Powered Fish Detection, 13 May 2025, PREPRINT (Version 1) available at Research Square [https://doi.org/10.21203/rs.3.rs-6538108/v1]
```

