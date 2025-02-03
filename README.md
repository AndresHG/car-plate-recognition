# Car plate recognition project

This is a Computer Vision project based on some Kaggle 
content I found plus my personal knowledge of this 
ecosystem. Please consider the project as a simple proof
of concept where you/we can learn about it.

Disclaimer: this is not using any best practices nor any
code design patterns. There are also no comments or proper
documentation since the purpose of this repo is just for
practical experience and trying some stuff.

## Instructions

First of all, Yolo folder and data files are saved in
Google Drive due to size concerns. Let's download them
from the link and move ahead. This is 
not a good practice at all, but I just wanted to simplify
the process as much as possible.

Link with the data: [HERE](https://drive.google.com/file/d/1FR7wqnh1Y0wcnq3HvoNykbkSnITmb1uW/view?usp=drive_link)

Please put the zip file in the root folder of the project
(at the same level of `video_detection.py` file) and the
let's proceed unzipping the file:

```shell
# If you don't have unzip already installed, you can run:
# sudo apt-get install unzip
unzip car-plate-recognition-assets.zip
```

Now, let's install the environment. This one is based on
Conda environments (I am now using Poetry more but...). I
will assume that you have it already downloaded so you
should be able to run:

```shell
conda env create --name car-plate-recognition --file=torch-env.yml
```

Now you can activate the environment and start playing
around:

```shell
conda activate car-plate-recognition
```

## Playing and training the model

There is a notebook called `yolo-car-plate-recognition.ipynb`
where you can use the model and interact with it, and there
are also 2 cells commented to train the model:

```shell
# !python ./yolov5/train.py --data ./data/data.yaml --cfg ./yolov5/models/yolov5s.yaml --batch-size 8 --name Model --epochs 100

# !python ./yolov5/export.py --weight ./yolov5/runs/train/Model3/weights/best.pt --include torchscript onnx
```

The model is already trained so you should not worry about 
it, but I am just sharing in case you want to feed new data
or re-train with new set of parameters.

Lastly, in order to generate the video with CPU or GPU
you can run the script called `video-detection.py`. There
is a specific line where you can change this:

```python
# GENERAL VARS
SAVE_VIDEO = True
DEVICE="cpu"  # DEVICE="gpu"
```

I hope this has been interesting and everything worked 
as expected!

BY!