# Pedestrian Crossing Intention Prediction  
  
## Notification  

**Predicting Pedestrian Crossing Intention with Feature Fusion and Spatio-Temporal Attention.**  

<p align="center">
<img src="model.png" alt="Our proposed model" align="middle" width="800"/>
</p>

Paper in ArXiv: https://arxiv.org/pdf/2104.05485v1.pdf (submitted to IROS 2021)  

This work improves the existing pedestrian crossing intention prediction method and achieves the latest state-of-the-art performance.    

This work is heavily relied on the pedestrian action prediction benchmark: `Kotseruba, Iuliia, Amir Rasouli, and John K. Tsotsos. "Benchmark for Evaluating Pedestrian Action Prediction." In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pp. 1258-1268, 2021.`

## Environment 

python = 3.8  
tensorflow-gpu = 2.2   
numpy, opencv, PIL, matplotlib, etc  
CPU:i7-6700K, GPU:RTX-2070super  

## Dataset Preparation  

Download the [JAAD Annotation](https://github.com/ykotseruba/JAAD) and put `JAAD` file to this project's root directory (as `./JAAD`).  

Download the [JAAD Dataset](http://data.nvision2.eecs.yorku.ca/JAAD_dataset/), and then put the video file `JAAD_clips` into `./JAAD` (as `./JAAD/JAAD_clips`).  

Copy `jaad_data.py` from the corresponding repositories into this project's root directory (as `./jaad_data.py`).  

In order to use the data, first, the video clips should be converted into images. This can be done using script `./JAAD/split_clips_to_frames.sh` following JAAD dataset's instruction.  

Above operation will create a folder called `images` and save the extracted images grouped by corresponding video ids in the `./JAAD/images `folder.  
```
./JAAD/images/video_0001/
				00000.png
				00001.png
				...
./JAAD/images/video_0002/
				00000.png
				00001.png
				...		
...
```
## Training   

Note: our model extracts the semantic mask via DeeplabV3 (you need download pretrained segmentation model [deeplabv3](http://download.tensorflow.org/models/deeplabv3_mnv2_cityscapes_train_2018_02_05.tar.gz) before training and put checkpoint file into this project's root directory (as `./deeplabv3_mnv2_cityscapes_train_2018_02_05.tar.gz`) so that the model can obtain the input semantic data).    

Use `train_test.py` script with `config_file`:
```
python train_test.py -c <config_file>
```

All config_files are saved in `./config_files` and you can review all offered model configs in `./config_files/config_list.yaml` and all offered model architectures in `./model_imgs` corresponding to configs.  

For example, to train MASK-PCPA model run:  

```
python train_test.py -c config_files/ours/MASK_PCPA_jaad_2d.yaml
```  

The script will automatially save the trained model weights, configuration file and evaluation results in `models/<dataset>/<model_name>/<current_date>/` folder.

See comments in the `configs_default.yaml` and `action_predict.py` for parameter descriptions.

Model-specific YAML files contain experiment options `exp_opts` that overwrite options in `configs_default.yaml`.  


## Test saved model  

To re-run test on the saved model use:  

```
python test_model.py <saved_files_path>
```

For example:  
```
python test_model.py models/jaad/MASK_PCPA/xxxx/
```  

You can download our pretrained models from [Google Drive (to do)](https://drive.google.com/drive/)     
or [BaiduDisk](https://pan.baidu.com/s/1GTvrcfe4a34sfwydVSQDqg) (password: v90h) for testing.    

## TODO Lists

- [x] Readme Completion
- [x] Pretrained Model
- [ ] Support PIE Dataset
