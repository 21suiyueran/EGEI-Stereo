<!--
 * @Author: zhaofengan
 * @Date: 2024-03-03 21:53:03
 * @LastEditTime: 2024-07-26 23:05:06
 * @FilePath: \EGEI-Stereo\README.md
 * @Description: 
 * 
-->
<h1>Edge-Guided Fusion and Motion Augmentation for Event-Image Stereo</h1>

This repository contains the source code for our ECCV 2024 paper:

**Edge-Guided Fusion and Motion Augmentation for Event-Image Stereo**


<img src="EGEIStereo.png" width="80%" height="80%">


## 🔧 Requirements
The code has been tested with PyTorch 1.11 and Cuda 11.3
```Shell
conda env create -f environment.yaml
conda activate egeistereo
```

## 💾 Required Data
* To evaluate/train EGEI-stereo, you will need to download the [MVSEC](https://daniilidis-group.github.io/mvsec/download/) (Includes Indoor flying 1, Indoor Flying 2 & Indoor Flying 3) dataset, and convert the depths to left-view sub-pixel disparities by following instructions on ["Learning an event sequence embedding for dense event-based deep stereo by Tulyakov, S. and Fleuret, F. and Kiefel, M. and Gehler, P. and Hirsch, M."](https://github.com/tlkvstepan/event_stereo_ICCV2019).
<br>

* For your convenience, we prepare an [download link](https://www.dropbox.com/scl/fi/jfglbs4izqtz73pcpmysz/MVSEC.zip?rlkey=oj9gspx2m20oql65yi9qh1grc&dl=0) with the expected directory structure. Please download and unzip it to the current directory.

By default, `stereo_datasets.py` will search for the dataset in the following locations. 

```Shell
├── MVSEC
    ├── indoor_flying_1
        ├── disparity_image
        ├── event0
        ├── event1
        ├── image0
        ├── image1
        ├── timestamps.txt
    ├── indoor_flying_2
        └── ...
    ├── indoor_flying_3
        └── ...
```


## 🤖 Demo
The pre-trained models are in the `pre-trained` folder.

You can demo a trained model on the MVSEC dataset. To predict stereo for split 1, run
```Shell
python demo.py --path MVSEC --restore_ckpt pre-trained/EGEI-stereo_split1.pth --split 1 --mixed_precision --mode demo
```
Or for split 3:
```Shell
python demo.py --path MVSEC --restore_ckpt pre-trained/EGEI-stereo_split3.pth --split 3 --mixed_precision --mode demo
```
The visualization results will be saved in the `demo_visualization` folder.


## 💻 Evaluation

To evaluate a trained model on the test set for split 1, run
```Shell
python evaluate_stereo.py --path MVSEC --restore_ckpt pre-trained/EGEI-stereo_split1.pth --split 1 --mixed_precision --mode test
```
Or for split 3:
```Shell
python evaluate_stereo.py --path MVSEC --restore_ckpt pre-trained/EGEI-stereo_split3.pth --split 3 --mixed_precision --mode test
```

## 🚀 Training

Our model is trained on a single NVIDIA RTX 3080Ti GPU using the following command. Training logs will be written to `runs/` which can be visualized using tensorboard.
For split 1:
```Shell
python train_stereo.py --path MVSEC --split 1 --train_iters 12 --valid_iters 12 --mixed_precision --mode train
```
For split 3:
```Shell
python train_stereo.py --path MVSEC --split 3 --train_iters 12 --valid_iters 12 --mixed_precision --mode train
```

## 🎓 Citation
If you find this code useful for your research, please consider citing our paper:
```
@inproceedings{EGEI-Stereo,
  title={Edge-Guided Fusion and Motion Augmentation for Event-Image Stereo},
  author={Zhao, Fengan and Zhou, Qianang and Xiong, Junlin},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2024}
}
```
## 💡 Acknowledgement
Thanks to the inspirations and codes from the following excellent open-source projects:
[RAFT-Stereo](https://github.com/princeton-vl/RAFT-Stereo), [TEED](https://github.com/xavysp/TEED), [EFNet](https://github.com/AHupuJR/EFNet),[SCSNet](https://github.com/Chohoonhee/SCSNet)
