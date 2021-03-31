#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 15:50:37 2020

@author: asabater
"""

import prediction_utils
import argparse

from dataset_scripts.ntu120_utils.triplet_ntu_callback import get_ntu_one_shot_triplets


# python demo_ntu_one_shot_benchmark.py --path_model './pretrained_models/ntu_benchmark_model/'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_model', type=str, help='Path of the model to be evaluated')
    args = parser.parse_args()
    
    model, model_params = prediction_utils.load_model(args.path_model, False)
	 
    with open(model_params['eval_ntu_one_shot_eval_anchors_file'], 'r') as f: one_shot_eval_anchors = f.read().splitlines()
    with open(model_params['eval_ntu_one_shot_eval_set_file'], 'r') as f: one_shot_eval_set = f.read().splitlines()
    
    acc_euc, acc_cos, _ = get_ntu_one_shot_triplets(model, model_params,
                              one_shot_eval_anchors, one_shot_eval_set, triplets=[],
                              in_memory_callback=False, cached_anchors=None, cached_targets=None)

    print(' * NTU one-shot evaluation: euclidean {:.4f} | cosine {:.4f}'.format(acc_euc, acc_cos))
    

if __name__ == '__main__':
    main()

