# Exploring the Point Feature Relation on Point Cloud for Multi-View Stereo

## Introduction
PFR-MVSNet proposes multiple point feature learning modules based on the point cloud structure for learning-based MVS.  Our method captures and dynamically learns the structural features of the scene implied in the point cloud.  Our method also adaptively divides the similarity region of structural features and learns point features by point transformer encoder for the estimated depth map.

## How to use

### Environment
The environment requirements are listed as follows:
- Pytorch 1.0.1 
- CUDA 9.0 
- CUDNN 7.4.2
- GCC5

### Installation
* Check out the source code 

    ```git clone https://github.com/Roy-Sky/PFR-MVSNet```
* Install dependencies 

    ```bash install_dependencies.sh```
* Compile CUDA extensions 

    ```bash compile.sh```

### Training
* Download the preprocessed [DTU training data](https://drive.google.com/file/d/1eDjh-_bxKKnEuz5h-HXS7EDJn59clx6V/view) from [MVSNet](https://github.com/YoYo000/MVSNet) and unzip it to ```data/dtu```.
* Train the network

    ```python pointmvsnet/train.py --cfg configs/dtu_wde3.yaml```
  
  You could change the batch size in the configuration file according to your own pc.

### Testing
* Download the [rectified images](http://roboimagedata2.compute.dtu.dk/data/MVS/Rectified.zip) from [DTU benchmark](http://roboimagedata.compute.dtu.dk/?page_id=36) and unzip it to ```data/dtu/Eval```.
* Test with your own model

    ```python pointmvsnet/test.py --cfg configs/dtu_wde3.yaml```
    
* Test with the pretrained model

    ```python pointmvsnet/test.py --cfg configs/dtu_wde3.yaml TEST.WEIGHT outputs/dtu_wde3/model_pretrained.pth```

### Depth Fusion
PointMVSNet generates per-view depth map. We need to apply depth fusion ```tools/depthfusion.py``` to get the complete point cloud. Please refer to [MVSNet](https://github.com/YoYo000/MVSNet) for more details.
    

### Acknowledgments

We borrow some code from PointMVSNet and DH-RMVSNet. We thank the authors for releasing the source code.