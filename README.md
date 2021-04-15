# Skeleton-based One-shot Action Recognition

This repository contains the code to train and evaluate the work presented in the article [One-shot action recognition in challenging therapy scenarios](https://arxiv.org/abs/2102.08997).

## Download pre-trained models

Download the desired models used in the paper and store them under `./pretrained_models/`.

* [NTU Bnechmark model](https://drive.google.com/file/d/18tif1Hj0ayXdnsbMocjDXCbKgWLAPPn7/view?usp=sharing)
* [Therapies model](https://drive.google.com/file/d/1uMO-AMU6D68lTj8z2sn9aQWRb7O7oX0C/view?usp=sharing)


## NTU-Dataset

NTU data must be downloaded and stored under `./datasets/NTU-120/raw_npy/`.


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


## Speed evaluation

Execute the following commands to test the action recognition speed on the NTU dataset.

```
python demo_speed_ntu.py --use_ntu --use_gpu --test_online --test_offline --max_clips 1000 --path_model './pretrained_models/ntu_benchmark_model/' --path_ntu_anns './ntu_annotations/one_shot_aux_set_full.txt' 
python demo_speed_ntu.py --use_ntu --test_online --test_offline --max_clips 1000 --path_model './pretrained_models/ntu_benchmark_model/' --path_ntu_anns './ntu_annotations/one_shot_aux_set_full.txt' 
```




** Due to privacy constraints, the therapy dataset is currently not public, so its evaluation is not available
