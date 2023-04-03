# Skeleton-based One-shot Action Recognition

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/one-shot-action-recognition-towards-novel/one-shot-3d-action-recognition-on-ntu-rgbd)](https://paperswithcode.com/sota/one-shot-3d-action-recognition-on-ntu-rgbd?p=one-shot-action-recognition-towards-novel)

[[Paper](https://arxiv.org/abs/2102.08997)] [[Supplementary video](https://drive.google.com/file/d/1NmY0vw78YwJ0ciKlUKwGU6Wrl9XVkTbO/view?usp=sharing)]

This repository contains the code to train and evaluate the work presented in the article [One-shot action recognition in challenging therapy scenarios](https://arxiv.org/abs/2102.08997).

```
@InProceedings{Sabater_2021_CVPR,
    author    = {Sabater, Alberto and Santos, Laura and Santos-Victor, Jose and Bernardino, Alexandre and Montesano, Luis and Murillo, Ana C.},
    title     = {One-Shot Action Recognition in Challenging Therapy Scenarios},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2021},
    pages     = {2777-2785}
}
```


## Download pre-trained models

Download the desired models used in the paper and store them under `./pretrained_models/`.

* [NTU Bnechmark model](https://unizares-my.sharepoint.com/:u:/g/personal/asabater_unizar_es/EamXVPDPFtFKtn1z26n5qhMBRMGWS8mDSXL-wfORQoHdLQ?e=YgLlcD)
* [Therapies model](https://unizares-my.sharepoint.com/:u:/g/personal/asabater_unizar_es/EVeQYXBP5dZNv0pD3-2485MB47RrMz7tA5KnfdJVnJLCqA?e=mNjK7C)


## Datasets

[NTU-120 dataset](https://rose1.ntu.edu.sg/dataset/actionRecognition/) must be downloaded and stored under `./datasets/NTU-120/raw_npy/`.

[Therapy dataset](https://doi.org/10.5281/zenodo.4700564) (pickle files) must be downloaded and stored under `./datasets/therapies_dataset/`.


## Python dependencies

Project tested with the following dependencies:

 * python 3.6
 * tensorflow 2.3.0
 * Keras 2.3.1
 * keras-tcn 3.1.0
 * scikit-learn 0.22.2
 * scipy 1.4.1
 * pandas 1.0.3


## NTU Benchmark evaluation

To evaluate the accuracy of our approach on the NTU-120 One-shot action recognition challenge execute:

`python demo_ntu_one_shot_benchmark.py --path_model './pretrained_models/ntu_benchmark_model/'`


## Therapies evaluation

Following command will read the best parameters to execute the final classification for each set-up (distance metric, one-shot, few-shot, and with dynamic threshold):

`python demo_therapies_benchmark.py --path_model ./pretrained_models/therapies_model_7/`

Following command will re-calculate and store the best parameters for each set-up (distance metric, one-shot, few-shot, and with dynamic threshold):

`python curves_comparison.py --path_model ./pretrained_models/therapies_model_7/ --force_all`


## Speed evaluation

Execute the following commands to test the action recognition speed in the therapy dataset:

```
python demo_speed.py --use_therapies --use_gpu --test_online --test_offline --max_clips 1000 --path_model './pretrained_models/ntu_benchmark_model/' --path_ntu_anns './ntu_annotations/one_shot_aux_set_full.txt' 
python demo_speed.py --use_therapies --test_online --test_offline --max_clips 1000 --path_model './pretrained_models/ntu_benchmark_model/' --path_ntu_anns './ntu_annotations/one_shot_aux_set_full.txt' 
```

Execute the following commands to test the action recognition speed in the NTU-120 dataset:

```
python demo_speed.py --use_ntu --use_gpu --test_online --test_offline --max_clips 1000 --path_model './pretrained_models/ntu_benchmark_model/' --path_ntu_anns './ntu_annotations/one_shot_aux_set_full.txt' 
python demo_speed.py --use_ntu --test_online --test_offline --max_clips 1000 --path_model './pretrained_models/ntu_benchmark_model/' --path_ntu_anns './ntu_annotations/one_shot_aux_set_full.txt' 
```


