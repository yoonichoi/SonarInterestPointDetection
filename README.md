# SonarInterestPointDetection

## Installation

### Requirements
- Python 3 >= 3.5
- PyTorch >= 1.1
- OpenCV >= 3.4
- NumPy >= 1.18
```
conda create --name sonar-ip python=3.6
conda activate sonar-ip
pip install -r requirements.txt
```

## Repository Structure

    .
    ├── data                   
    |   ├──demo-synthethic-shapes
    |   └──sonar-data
    |   |   └──test
    |   |   |   ├──img
    |   |   |   └──pts
    |   |   └──train
    |   |   |   ├──img
    |   |   |   └──pts
    |   |   └──val
    |   |   |   ├──img
    |   |   |   └──pts
    |   └──annotation_to_npy.py
    ├── demo                   
    |   ├──pretrained-superpoint.pth
    |   └──sampled_trainer_notebook.ipynb
    ├── src                   
    |   ├──data
    |   |   ├──dataloader.py
    |   |   └──dataset.py
    |   ├──loss.py
    |   ├──superpoint.py
    |   ├──trainer.py
    |   └──utils.py             
    ├── README.md
    └── requirements.txt

## How to run the file

```.py``` files are available, however we advise you to run the demo notebook (```.\demo\sample_trainer_notebook.py```) in your Google Colab instance.




## About the data

Data folder structure should follow above structure in order to run the training instance without running into any error. Data should be split into test, train, and validation folders each comprising ```img``` and ```pts``` folders. ```img``` should contain original ```.png``` files of your sonar image data, whereas ```pts``` should contain ```.npy``` files of your annotated interest points from the original image. We have used [MakeSense.ai](https://www.makesense.ai/) in order to manual annotate the interest points of the image files. Once you export ```.csv``` of your annotations, you can refer to ```.\data\annotation_to_npy.py``` file for the ```.npy``` conversion. 
### Example of ```img\*.png``` file
<img src="https://github.com/yoonichoi/SonarInterestPointDetection/blob/main/data/sonar-data/test/img/1010.png?raw=true" width="300">

### Example of ```pts\*.npy``` file
``` 
array ([[201, 404],
       [243, 439],
       [317, 415]])
```
