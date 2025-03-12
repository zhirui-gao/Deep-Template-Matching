# Deep-Template-Matching 🌟  
**Learning Accurate Template Matching with Differentiable Coarse-to-fine Correspondence Refinement**

🚀 Official implementation of Deep-Template-Matching (Learning Accurate Template Matching with Differentiable Coarse-to-fine Correspondence Refinement) using pytorch ([pytorch-lightning](https://github.com/Lightning-AI/lightning))  


✨ This paper has published in [CVMJ 2024](https://www.springer.com/journal/41095) and can be founded [here](https://link.springer.com/article/10.1007/s41095-023-0333-9). If you find this repository useful to your research or work, it is really appreciated to star this repository and cite our paper 📚. :sparkles:

---

## 🎯 Abstract  
Template matching is a fundamental task in computer vision and has been studied for decades. It plays an essential role in the manufacturing industry for estimating the poses of different parts, facilitating downstream tasks such as robotic grasping. Existing works fail when the template and source images are in different modalities, cluttered backgrounds or weak textures. They also rarely consider geometric transformations via homographies, which commonly existed even for planar industrial parts. To tackle the challenges, we propose an accurate template matching method based on differentiable coarse-to-fine correspondence refinement. Considering the domain gap between the mask template and the grayscale image, we leverage an edge-aware module to eliminate the difference for robust matching. Based on coarse correspondences with novel structure-aware information by transformers, an initial warping transformation is estimated and performed as a preliminary result. After the initial alignment, we execute a refinement network on reference and aligned images to obtain sub-pixel level correspondences and thus obtain the final geometric transformation. Comprehensive evaluations show that our method significantly outperforms state-of-the-art methods and baselines, with good generalization abilities and visually plausible results even on unseen real data.

---

## 🔑 Introduction
we propose an accurate template matching method based on differentiable coarse-to-fine correspondence refinement. Considering the domain gap between the mask template and the grayscale image, we leverage an edge-aware module to eliminate the difference for robust matching. Based on coarse correspondences with novel structure-aware information by transformers, an initial warping transformation is estimated and performed as a preliminary result. After the initial alignment, we execute a refinement network on reference and aligned images to obtain sub-pixel level correspondences and thus obtain the final geometric transformation. 

![image](https://github.com/zhirui-gao/Deep-Template-Matching/blob/master/teaser.png)


## ✨ Installation
```
pip -r requirements.txt
```

We provide the datasets used in our paper. [Download link](https://drive.google.com/drive/folders/1Mu9QdnM5WsLccFp0Ygf7ES7mLV-64wRL?usp=sharing) to
- Assembled hole dataset
- Steel dataset

## Run Deep-Template-Matching demo
An example is ```given in test_demo.py```.
The test images are in ```./data/test_case``` and the pretraind weight file can be downloaded from [here](https://drive.google.com/file/d/1__Az9VqbLy28TAosnHpNJJLrQEGQ4pAJ/view?usp=drive_link).




## ✨ Training
run command ```./scripts/train.sh```. The traning files are saved in direactory ```./logs```.
### 0. Data preparation
Please modify the paths of the training dataset  in the (```./config/Synthetic_train.py```). 
We have prepared a standard data format in the folder(```./data/train_data```)
  
Synthetic_train.py:
 ```angular2html  
    TRAIN_BASE_PATH = './data/train_data'
 ```


### 1. Training steps
**We use a two-stage training method.**（Modify configuration parameters in ```./src/config/default.py```）  
In the coarse stage, we only train the coarse network until convergence(about 10-20 epochs):
```angular2html
_CN.TM.MATCH_COARSE.TRAIN_STAGE = 'only_coarse'
```
In the fine stage,  modify  the ckpt_path in ```train.py```
```angular2html
parser.add_argument(
        '--ckpt_path', type=str, default='', # the path of coarse ckpt
        help='pretrained checkpoint path')
```

 We train the whole network until convergence(about 10-20 epochs)::
```angular2html
_CN.TM.MATCH_COARSE.TRAIN_STAGE = 'whole'
```
The detail files of training are saved in the ```./logs``` folder

### 2. Edge detetion 
- If the edge of the test data is easy to detect, we recommend
    ```angular2html
    _CN.TM.MATCH_COARSE.USE_EDGE = True   #better generalization, as default
    ```
- otherwise
    ```angular2html
    _CN.TM.MATCH_COARSE.USE_EDGE = False
    ```

### 3. Additional configuration parameters 
```./src/config/default.py```
- Use online data augmentation:
    ```angular2html
    _CN.DATASET.AUGMENTATION_TYPE = 'None'
    ```
    otherwise
    ```angular2html
    _CN.DATASET.AUGMENTATION_TYPE = 'mobile_myself'
    ```
- Save Plots of matching images to the training file using tensorboard:
    ```angular2html
    _CN.TRAINER.SAVE_PLOTS_VAL = True
    _CN.TRAINER.SAVE_PLOTS_TRAIN = False
    ```
    otherwise
    ```angular2html
     _CN.TRAINER.SAVE_PLOTS_VAL = False
    _CN.TRAINER.SAVE_PLOTS_TRAIN = False
    ```

### 4. image resize
- All images are resized to [480, 640] (h,w), and we set the max number of query points is 128.
- If you want change the size of images,please change ```Resize = [512, 512] # h,w``` in ```./src/lightning/data.py```.
- The image size is not recommended to be too small, otherwise the matching pair will decline seriously.


## Test
0. Change the path of the test dataset in ```./config/Synthetic_test.py```
 ```angular2html  
    TEST_BASE_PATH = './data/train_data'
 ```
1. Run command ```./scripts/test.sh```

## 🚀 Demos

### Single-object demo

![Output sample](https://github.com/zhirui-gao/Deep-Template-Matching/blob/master/single_object.gif)

### Multi-objects demo
![s](https://github.com/zhirui-gao/Deep-Template-Matching/blob/master/multi_object.gif)

## 📚 Reference 
If you find this code useful for your research, please use the following BibTeX entry.
```
@article{gao2024learning,
  title={Learning accurate template matching with differentiable coarse-to-fine correspondence refinement},
  author={Gao, Zhirui and Yi, Renjiao and Qin, Zheng and Ye, Yunfan and Zhu, Chenyang and Xu, Kai},
  journal={Computational Visual Media},
  volume={10},
  number={2},
  pages={309--330},
  year={2024},
  publisher={Springer}
}
```

## Acknowledgements
- [LoFTR](https://github.com/zju3dv/LoFTR)  
- [SuperGlue](https://github.com/magicleap/SuperGluePretrainedNetwork)
- [pidinet](https://github.com/zhuoinoulu/pidinet)
- [PointDSC](https://github.com/XuyangBai/PointDSC)
